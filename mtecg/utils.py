import os.path as op
import pandas as pd
from glob import glob
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
    image_extension: str = "jpg",
) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path).drop(columns=["Unnamed: 0"])
    # Lowercase column names for consistency.
    dataframe.columns = map(str.lower, dataframe.columns)

    if constants.cut_column_name in dataframe.columns:
        dataframe = dataframe[dataframe[constants.cut_column_name] != 1].reset_index(drop=True)
    if constants.impute_column_name in dataframe.columns and drop_impute:
        dataframe = dataframe[dataframe[constants.impute_column_name] != 1].reset_index(drop=True)

    dataframe[constants.path_column_name] = dataframe.apply(
        lambda row: op.join(
            image_dir,
            f"{row[constants.filename_column_name]}_{row[constants.run_number_column_name]}.{image_extension}",
        ),
        axis=1,
    )

    # Merge with the available images.
    image_dataframe = pd.DataFrame(
        glob(op.join(image_dir, f"*.{image_extension}")), columns=[constants.path_column_name]
    )
    # Replace \\ with / to avoid path issues.
    image_dataframe[constants.path_column_name] = image_dataframe[constants.path_column_name].apply(
        lambda x: x.replace("\\", "/")
    )
    dataframe = dataframe.merge(image_dataframe, on=constants.path_column_name)

    # Rename columns.
    dataframe.rename(columns=constants.COLUMN_RENAME_MAP, inplace=True)

    # Scale age column.
    dataframe[constants.age_column_name] = dataframe[constants.age_column_name] / 100
    dataframe[constants.age_column_name] = dataframe[constants.age_column_name].astype(float)

    # Categorize LVEF.
    dataframe[constants.lvef_label_column_name] = dataframe[constants.lvef_label_column_name].apply(
        lambda lvef: categorize_lvef(lvef, lvef_threshold)
    )

    # Generate split column.
    if do_split:
        dataframe = dataframe.apply(get_split_by_year, axis=1)
    return dataframe


def merge_dict_list(dict_list_1, dict_list_2):
    merged_dict_list = []
    for dict_1, dict_2 in zip(dict_list_1, dict_list_2):
        merged_dict = {**dict_1, **dict_2}
        merged_dict_list.append(merged_dict)
    return merged_dict_list
