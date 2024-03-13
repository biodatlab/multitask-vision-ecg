import os.path as op
import numpy as np
import pandas as pd
from glob import glob
from typing import List

import mtecg.constants as constants


def categorize_lvef(lvef, threshold: int = 50):
    return 1 if lvef <= threshold else 0


def get_split_by_year(row):
    year = row[constants.year_column_name]
    if year == 2015:
        row[constants.split_column_name] = "old_valid"
    elif year == 2016:
        row[constants.split_column_name] = "old_test"
    elif (year != 2015) and (year != 2016) and (year < 2017):
        row[constants.split_column_name] = "old_train"
    elif (row[constants.train_column_name] == 1) and (year >= 2017):
        row[constants.split_column_name] = "new_train"
    elif row[constants.dev_column_name] == 1:
        row[constants.split_column_name] = "new_valid"
    return row


def load_ecg_dataframe(
    csv_path: str,
    image_dir: str,
    lvef_threshold: int = 50,
    do_split: bool = True,
    drop_impute: bool = False,
    imputer_dir: str = None,
    image_extension: str = "jpg",
    is_control_population: bool = False,
    return_lvef_40_column: bool = False,
) -> pd.DataFrame:

    if not is_control_population:
        dataframe = pd.read_csv(csv_path)
        if "Unnamed: 0" in dataframe.columns:
            dataframe.drop(columns=["Unnamed: 0"], inplace=True)
    else:
        dataframe = (
            pd.read_csv(csv_path)
            .drop(columns=[constants.path_column_name])
            .rename(columns={"save_path": constants.path_column_name})
        )
        # Set all labels to 0 for control population.
        dataframe[constants.scar_label_column_name] = 0
        dataframe["LVEF"] = 100

    # Lowercase column names for consistency.
    dataframe.columns = map(str.lower, dataframe.columns)

    if constants.cut_column_name in dataframe.columns:
        dataframe = dataframe[dataframe[constants.cut_column_name] != 1].reset_index(drop=True)
    if constants.impute_column_name in dataframe.columns and drop_impute:
        dataframe = dataframe[dataframe[constants.impute_column_name] != 1].reset_index(drop=True)

    # Get the available images dataframe.
    image_dataframe = pd.DataFrame(
        glob(op.join(image_dir, f"*.{image_extension}")), columns=[constants.path_column_name]
    )

    # The control population has a different format for the path column.
    if is_control_population:
        image_dataframe["name"] = image_dataframe[constants.path_column_name].apply(
            lambda x: op.splitext(op.basename(x))[0]
        )
        dataframe["name"] = dataframe[constants.path_column_name].apply(lambda x: op.splitext(op.basename(x))[0])
        dataframe = dataframe.merge(image_dataframe, on="name")
        dataframe.drop(columns=["name"], inplace=True)

    else:
        dataframe[constants.path_column_name] = dataframe.apply(
            lambda row: op.join(
                image_dir,
                f"{row[constants.filename_column_name]}_{row[constants.run_number_column_name]}.{image_extension}",
            ),
            axis=1,
        )
        # Replace \\ with / to avoid path issues.
        dataframe[constants.path_column_name] = dataframe[constants.path_column_name].apply(
            lambda x: x.replace('\\', '/')
        )
        image_dataframe[constants.path_column_name] = image_dataframe[constants.path_column_name].apply(
            lambda x: x.replace('\\', '/')
        )

        # # Merge with the available images.
        # dataframe = dataframe.merge(image_dataframe, on=constants.path_column_name)
        dataframe = dataframe.merge(image_dataframe, on=constants.path_column_name)

    # Rename columns.
    dataframe.rename(columns=constants.COLUMN_RENAME_MAP, inplace=True)

    # Scale age column if it is in the dataframe.
    if constants.age_column_name in dataframe.columns:
        dataframe[constants.age_column_name] = dataframe[constants.age_column_name] / 100
        dataframe[constants.age_column_name] = dataframe[constants.age_column_name].astype(float)

    # Categorize LVEF.
    # Also return the LVEF_40 column if return_lvef_40_column is True.
    if return_lvef_40_column:
        dataframe[constants.lvef_40_column_name] = dataframe[constants.lvef_label_column_name].apply(
            lambda lvef: categorize_lvef(lvef, 40)
        )

    dataframe[constants.lvef_label_column_name] = dataframe[constants.lvef_label_column_name].apply(
        lambda lvef: categorize_lvef(lvef, lvef_threshold)
    )

    if imputer_dir:
        import joblib

        imputer = joblib.load(op.join(imputer_dir, "imputer.joblib"))
        threshold_dict = joblib.load(op.join(imputer_dir, "imputer_threshold_dict.joblib"))

        clinical_feature_columns = constants.numerical_feature_column_names + constants.categorical_feature_column_names
        # Mark all imputed values as np.nan so that they can be imputed by the imputer.
        dataframe.loc[dataframe[constants.impute_column_name] == True, constants.imputed_feature_column_names] = np.nan
        # Impute the missing values.
        dataframe[clinical_feature_columns] = imputer.transform(dataframe[clinical_feature_columns])
        # Apply the thresholds to the imputed values so that they are either 0 or 1.
        dataframe = apply_thresholds(dataframe, threshold_dict)

    # Generate split column.
    if do_split:
        if "split" not in dataframe.columns:
            dataframe = dataframe.apply(get_split_by_year, axis=1)
    return dataframe


def merge_dict_list(dict_list_1, dict_list_2):
    merged_dict_list = []
    for dict_1, dict_2 in zip(dict_list_1, dict_list_2):
        merged_dict = {**dict_1, **dict_2}
        merged_dict_list.append(merged_dict)
    return merged_dict_list


def find_best_thresholds(
    dataframe: pd.DataFrame,
    imputed_column_names: List[str] = constants.imputed_feature_column_names,
):
    original_df = dataframe[dataframe[constants.impute_column_name] == False].reset_index()[imputed_column_names]
    impute_df = dataframe[dataframe[constants.impute_column_name] == True].reset_index()[imputed_column_names]

    # Try out different threshold for categorizing the imputed values into 0 or 1 so that the distribution of the imputed values is similar to the original values.
    # Store the result that has the smallest difference between the prevalence of the imputed values and the original values.
    # The threshold that gives the smallest difference is the threshold that gives the best imputation.
    # The threshold of each column is stored in a dictionary.
    best_threshold_dict = {}
    for imputed_column_name in imputed_column_names:
        for threshold in np.arange(0.1, 0.9, 0.01):
            imputed_df = pd.DataFrame(
                impute_df.values > threshold,
                columns=imputed_column_names,
                index=impute_df.index,
            )
            imputed_df = imputed_df.astype(int)
            diff = abs(
                original_df[imputed_column_name].sum() / len(original_df)
                - imputed_df[imputed_column_name].sum() / len(imputed_df)
            )
            if imputed_column_name not in best_threshold_dict:
                best_threshold_dict[imputed_column_name] = [round(threshold, 2), diff]
            else:
                if diff < best_threshold_dict[imputed_column_name][1]:
                    best_threshold_dict[imputed_column_name] = [round(threshold, 2), diff]
    return best_threshold_dict


def apply_thresholds(
    dataframe: pd.DataFrame,
    best_threshold_dict: dict,
    imputed_column_names: List[str] = constants.imputed_feature_column_names,
):
    impute_df = dataframe[dataframe[constants.impute_column_name] == True].reset_index()[imputed_column_names]
    for imputed_column_name in imputed_column_names:
        threshold = best_threshold_dict[imputed_column_name][0]
        imputed_df = pd.DataFrame(
            impute_df.values > threshold,
            columns=imputed_column_names,
            index=impute_df.index,
        )
        imputed_df = imputed_df.astype(int)
        dataframe.loc[dataframe[constants.impute_column_name] == True, imputed_column_name] = imputed_df[
            imputed_column_name
        ].values
    return dataframe


def load_ecg_dataframe_1d(
    csv_path: str,
    data_dir: str,
    file_extension: str = "",
    lvef_threshold: int = 50,
    do_split: bool = True,
    drop_impute: bool = False,
    imputer_dir: str = None,
    is_control_population: bool = False,
    return_lvef_40_column: bool = False,
) -> pd.DataFrame:

    if not is_control_population:
        dataframe = pd.read_csv(csv_path)
        if "Unnamed: 0" in dataframe.columns:
            dataframe.drop(columns=["Unnamed: 0"], inplace=True)
    else:
        dataframe = (
            pd.read_csv(csv_path)
            .drop(columns=[constants.path_column_name])
            .rename(columns={"save_path": constants.path_column_name})
        )
        # Set all labels to 0 for control population.
        dataframe[constants.scar_label_column_name] = 0
        dataframe["LVEF"] = 100

    # Lowercase column names for consistency.
    dataframe.columns = map(str.lower, dataframe.columns)

    if constants.cut_column_name in dataframe.columns:
        dataframe = dataframe[dataframe[constants.cut_column_name] != 1].reset_index(drop=True)
    if constants.impute_column_name in dataframe.columns and drop_impute:
        dataframe = dataframe[dataframe[constants.impute_column_name] != 1].reset_index(drop=True)

    # Get the available array data dataframe.
    array_dataframe = pd.DataFrame(
        glob(op.join(data_dir, f"*{file_extension}")), columns=[constants.path_column_name]
    )

    # The control population has a different format for the path column.
    if is_control_population:
        array_dataframe["name"] = array_dataframe[constants.path_column_name].apply(
            lambda x: op.splitext(op.basename(x))[0]
        )
        dataframe["name"] = dataframe[constants.path_column_name].apply(lambda x: op.splitext(op.basename(x))[0])
        dataframe = dataframe.merge(array_dataframe, on="name")
        dataframe.drop(columns=["name"], inplace=True)

    else:
        dataframe[constants.path_column_name] = dataframe.apply(
            lambda row: op.join(
                data_dir,
                f"{row[constants.filename_column_name]}_{row[constants.run_number_column_name]}{file_extension}",
            ),
            axis=1,
        )
        # Replace \\ with / to avoid path issues.
        dataframe[constants.path_column_name] = dataframe[constants.path_column_name].apply(
            lambda x: x.replace('\\', '/')
        )
        array_dataframe[constants.path_column_name] = array_dataframe[constants.path_column_name].apply(
            lambda x: x.replace('\\', '/')
        )
        # Merge with the available images.
        dataframe = dataframe.merge(array_dataframe, on=constants.path_column_name)

    # Rename columns.
    dataframe.rename(columns=constants.COLUMN_RENAME_MAP, inplace=True)

    # Scale age column if it is in the dataframe.
    if constants.age_column_name in dataframe.columns:
        dataframe[constants.age_column_name] = dataframe[constants.age_column_name] / 100
        dataframe[constants.age_column_name] = dataframe[constants.age_column_name].astype(float)

    # Categorize LVEF.
    # Also return the LVEF_40 column if return_lvef_40_column is True.
    if return_lvef_40_column:
        dataframe[constants.lvef_40_column_name] = dataframe[constants.lvef_label_column_name].apply(
            lambda lvef: categorize_lvef(lvef, 40)
        )

    dataframe[constants.lvef_label_column_name] = dataframe[constants.lvef_label_column_name].apply(
        lambda lvef: categorize_lvef(lvef, lvef_threshold)
    )

    if imputer_dir:
        import joblib

        imputer = joblib.load(op.join(imputer_dir, "imputer.joblib"))
        threshold_dict = joblib.load(op.join(imputer_dir, "imputer_threshold_dict.joblib"))

        clinical_feature_columns = constants.numerical_feature_column_names + constants.categorical_feature_column_names
        # Mark all imputed values as np.nan so that they can be imputed by the imputer.
        dataframe.loc[dataframe[constants.impute_column_name] == True, constants.imputed_feature_column_names] = np.nan
        # Impute the missing values.
        dataframe[clinical_feature_columns] = imputer.transform(dataframe[clinical_feature_columns])
        # Apply the thresholds to the imputed values so that they are either 0 or 1.
        dataframe = apply_thresholds(dataframe, threshold_dict)

    # Generate split column.
    if do_split:
        if "split" not in dataframe.columns:
            dataframe = dataframe.apply(get_split_by_year, axis=1)
    return dataframe
