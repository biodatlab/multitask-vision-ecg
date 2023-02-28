import numpy as np
import pandas as pd
from math import sqrt
from tqdm.auto import tqdm
from typing import List, Dict, Any
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix


from mtecg import (
    SingleTaskModel,
    MultiTaskClinicalCNNModel,
)
from mtecg.classifier import ECGClassifier
from mtecg.utils import merge_dict_list
import mtecg.constants as constants


def evaluate_from_dataframe(
    dataframe: pd.DataFrame,
    classifier: ECGClassifier,
    path_column_name=constants.path_column_name,
    scar_column_name=constants.scar_label_column_name,
    lvef_column_name=constants.lvef_label_column_name,
    categorical_feature_column_names=constants.categorical_feature_column_names,
    numerical_feature_column_names=constants.numerical_feature_column_names,
    is_control_population: bool = False,
    average: str = "weighted",
) -> (pd.DataFrame, pd.DataFrame):

    output_dict_list = []
    image_paths = list(dataframe[path_column_name])
    scar_labels = list(dataframe[scar_column_name])
    lvef_labels = list(dataframe[lvef_column_name])

    model_input_dict_list = [{"image": path} for path in image_paths]

    if isinstance(classifier.model, MultiTaskClinicalCNNModel):
        feature_lists = [list(dataframe[column_name]) for column_name in categorical_feature_column_names]
        feature_lists += [list(dataframe[column_name]) for column_name in numerical_feature_column_names]

        feature_array = np.array(feature_lists)
        column_names = categorical_feature_column_names + numerical_feature_column_names
        # Convert the numpy array to a list of dictionaries for each patient with keys based on the categorical and numerical feature column names
        rows, columns = feature_array.shape

        feature_dict_list = []
        for column in range(columns):
            feature_dict = {}
            for column_name, feature in zip(column_names, feature_array[:, column]):
                feature_dict[column_name] = feature
            feature_dict_list.append(feature_dict)

        model_input_dict_list = [
            {"image": path, "clinical_features": feature_dict}
            for path, feature_dict in zip(image_paths, feature_dict_list)
        ]

    for i, model_input_dict in enumerate(tqdm(model_input_dict_list)):
        output_dict = classifier.predict(**model_input_dict)
        if "scar" in classifier.task:
            output_dict["scar"]["label"] = scar_labels[i]
        if "lvef" in classifier.task:
            output_dict["lvef"]["label"] = lvef_labels[i]

        output_dict_list.append(output_dict)

    result_dataframe = convert_output_to_dataframe(output_dict_list)
    metrics_dataframe = calculate_metrics(
        result_dataframe,
        tasks=classifier.task,
        is_control_population=is_control_population,
        average=average,
    )
    return result_dataframe, metrics_dataframe


def convert_output_to_dataframe(result_dict_list: List[Dict[str, Any]]):
    if "scar" in result_dict_list[0]:
        scar_label_list = []
        scar_prediction_list = []
        scar_probability_list = []
    if "lvef" in result_dict_list[0]:
        lvef_label_list = []
        lvef_prediction_list = []
        lvef_probability_list = []

    for result_dict in result_dict_list:
        if "scar" in result_dict:
            scar_label_list.append(result_dict["scar"]["label"])
            scar_prediction_list.append(result_dict["scar"]["prediction"])
            scar_probability_list.append(result_dict["scar"]["probability"]["positive"])
        if "lvef" in result_dict:
            lvef_label_list.append(result_dict["lvef"]["label"])
            lvef_prediction_list.append(result_dict["lvef"]["prediction"])
            lvef_probability_list.append(result_dict["lvef"]["probability"]["positive"])

    final_result_dict = {}
    if "scar" in result_dict_list[0]:
        final_result_dict["scar_label"] = scar_label_list
        final_result_dict["scar_prediction"] = scar_prediction_list
        final_result_dict["scar_probability"] = scar_probability_list
    if "lvef" in result_dict_list[0]:
        final_result_dict["lvef_label"] = lvef_label_list
        final_result_dict["lvef_prediction"] = lvef_prediction_list
        final_result_dict["lvef_probability"] = lvef_probability_list

    result_dataframe = pd.DataFrame(final_result_dict)
    return result_dataframe


def calculate_metrics_per_task(
    result_dataframe,
    task: str,
    is_control_population: bool = False,
    average: str = "weighted",
):
    label_column_name = f"{task}_label"
    prediction_column_name = f"{task}_prediction"
    probability_column_name = f"{task}_probability"

    tn, fp, fn, tp = confusion_matrix(
        result_dataframe[label_column_name], result_dataframe[prediction_column_name]
    ).ravel()

    accuracy = accuracy_score(result_dataframe[label_column_name], result_dataframe[prediction_column_name])
    specificity = tn / (tn + fp)
    f1 = f1_score(result_dataframe[label_column_name], result_dataframe[prediction_column_name], average=average)
    fpr = fp / (fp + tn)

    metrics_dict = {
        "Accuracy": [accuracy],
        "Sensitivity": [None],
        "Specificity": [specificity],
        "F1": [f1],
        "AUC": [None],
        "FPR": [fpr],
        "FNR": [None],
    }

    if not is_control_population:
        auc = roc_auc_score(result_dataframe[label_column_name], result_dataframe[probability_column_name])
        sensitivity = tp / (tp + fn)
        fnr = fn / (tp + fn)
        metrics_dict["AUC"] = [auc]
        metrics_dict["Sensitivity"] = [sensitivity]
        metrics_dict["FNR"] = [fnr]

    metrics_dataframe = pd.DataFrame(metrics_dict)
    metrics_dataframe = metrics_dataframe.T
    return metrics_dataframe


def calculate_metrics(
    result_dataframe,
    tasks: List[str] = ["scar", "lvef"],
    is_control_population: bool = False,
    average: str = "weighted",
):
    metrics_dataframe = pd.DataFrame()
    for task in tasks:
        task_metrics_dataframe = calculate_metrics_per_task(result_dataframe, task, is_control_population, average)
        metrics_dataframe = pd.concat([metrics_dataframe, task_metrics_dataframe], axis=1)
    metrics_dataframe.columns = tasks
    return metrics_dataframe


def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC**2) + (N2 - 1) * (Q2 - AUC**2)) / (N1 * N2))
    lower = AUC - 1.96 * SE_AUC
    upper = AUC + 1.96 * SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, upper)
