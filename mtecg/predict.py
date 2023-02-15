import numpy as np
from tqdm.auto import tqdm
import pathlib
from pathlib import Path
import pandas as pd
import os.path as op

# from fastbook import *
from .io import read_pdf_to_image
from torchvision import transforms as T
from PIL import Image
import torch
import torch.nn.functional as F


# plt = platform.system()
# if plt in ("Linux", "Darwin"):
#     pathlib.WindowsPath = pathlib.PosixPath


# def load_learner_path(model_path: str):
#     """
#     Load FastAI learner from a given path
#     """
#     assert op.exists(model_path), "Trained model does not exist."
#     model_path = Path(model_path)
#     learner = load_learner(model_path)  # load learner
#     return learner


# def load_multitask_model(model_path: str):
#     """
#     Load Pytorch Lightning multitask model from a given state dict path
#     """
#     assert op.exists(model_path), "Trained model does not exist."
#     from models import MultiTaskModel

#     model = MultiTaskModel()
#     model.load_state_dict(torch.load(model_path))
#     return model


# def predict_ecg(learner, path: str):
#     """
#     Predict ECG from a given path (can be PNG or PDF file).
#     Return a class and probability of prediction.
#     >>> pred, pred_proba = predict_ecg(path)  # path to PNG or PDF file
#     """
#     suffix = str(Path(path).suffix).lower()
#     if suffix == ".png":
#         prediction = learner.predict(path)
#     elif suffix == ".pdf":
#         img = read_pdf_to_image(path)
#         prediction = learner.predict(np.array(img))
#     else:
#         print("Please provide a PNG or PDF file for classification.")
#         return (None, None)
#     pred, _, pred_proba = prediction
#     return pred, [float(e) for e in list(pred_proba)]


# def predict_ecg_list(learner, paths: list):
#     """
#     Given learner and a list of paths to PDF files,
#     predict ECG and return a list of class and probability of prediction.
#     """
#     predictions = []
#     for path in tqdm(paths):
#         predictions.append(predict_ecg(learner, path))
#     y_pred = [p[0] for p in predictions]
#     y_pred_proba = [p[1][0] for p in predictions]
#     return y_pred, y_pred_proba


def evaluate(
    y_true: np.array,
    y_pred: np.array,
    y_pred_proba: np.array,
    map_dict: dict = {"Normal": 0, "Abnormal": 1},
):
    """
    Evaluate model performance
    """
    from sklearn.metrics import (
        precision_recall_fscore_support,
        accuracy_score,
        roc_auc_score,
    )

    y_true_binary = np.array([map_dict.get(p) for p in y_true])
    auc_roc = roc_auc_score(y_true_binary, y_pred_proba)
    accuracy = accuracy_score(y_true, y_pred)

    print("AUC : ", auc_roc)
    print("Accuracy : ", accuracy)
    print(
        "Weighted Precision, Recall, F1-score: ",
        precision_recall_fscore_support(y_true, y_pred, average="weighted"),
    )
    print("Precision, Recall, F1-score: ", precision_recall_fscore_support(y_true, y_pred))


def predict_from_dl(model, test_loader):
    """
    Given model and test dataloader, predict output class and probability
    and corresponding labels

    Examples
    ========
    >>> y_pred, y_true = predict(model_inf, test_loader)
    >>> precision_recall_fscore_support(y_true["scar"], y_pred["scar"])
    >>> accuracy_score(y_true["scar"], y_pred["scar"])
    >>> roc_auc_score(y_true["scar"], y_pred["scar_proba"])

    >>> precision_recall_fscore_support(y_true["lvef"], y_pred["lvef"])
    >>> accuracy_score(y_true["lvef"], y_pred["lvef"])
    >>> roc_auc_score(y_true["scar"], y_pred["lvef_proba"])
    """
    model.eval()
    with torch.no_grad():
        y_pred = {"scar": [], "lvef": [], "scar_proba": [], "lvef_proba": []}
        y_true = {"scar": [], "lvef": []}

        for imgs, labels in tqdm(test_loader):
            pred = model(imgs)
            # scar prediction
            y_true["scar"].extend(labels["scar"].tolist())
            pred_scar_proba = F.softmax(pred["scar"], dim=1)
            pred_scar = pred_scar_proba.argmax(dim=1).tolist()
            y_pred["scar"].extend(pred_scar)
            y_pred["scar_proba"].extend(pred_scar_proba[:, 1].tolist())

            # lvef prediction
            y_true["lvef"].extend(labels["lvef"].tolist())
            pred_lvef_proba = F.softmax(pred["lvef"], dim=1)
            pred_lvef = pred_lvef_proba.argmax(dim=1).tolist()
            y_pred["lvef"].extend(pred_lvef)
            y_pred["lvef_proba"].extend(pred_lvef_proba[:, 1].tolist())
    return y_pred, y_true


def predict_single_ecg(path, model):
    """Predict a given ECG in PNG format"""
    valid_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    model.eval()
    with torch.no_grad():
        img = Image.open(path)
        pred = model(valid_transform(img).unsqueeze(0))
        pred_scar = F.softmax(pred["scar"]).ravel()
        pred_lvef = F.softmax(pred["lvef"]).ravel()
        prediction = {
            "scar_proba": pred_scar.tolist(),
            "scar": int(pred_scar.argmax()),
            "lvef40_proba": pred_lvef.tolist(),
            "lvef": int(pred_lvef.argmax()),
        }
    return prediction


@torch.no_grad()
def predict_from_df(
    my_model,
    dataframe,
    input_transforms,
    image_size=(224, 224),
    lvef_threshold=50,
    device="cuda",
):
    """
    Given my_model and a dataframe, predict output class and probability
    and corresponding labels
    """

    def convert_tensor_list(tensor_list: list):
        return [t.cpu().numpy() for t in tensor_list]

    scar_true = dataframe.scar_cad.values.tolist()
    lvef_true = dataframe.LVEF.values.tolist()
    lvef_true = [1 if x <= lvef_threshold else 0 for x in lvef_true]
    paths = dataframe.path.values.tolist()
    # labels = dataframe.label.values.tolist()

    scar_preds = []
    lvef_preds = []
    scar_probs = []
    lvef_probs = []
    for i, row in tqdm(dataframe.iterrows()):
        img = Image.open(row["path"]).convert("RGB")
        img = input_transforms(image=np.array(img))["image"]
        pred = my_model(img.view(1, 3, image_size[0], image_size[1]).to(device))

        # scar prediction
        pred_scar_proba = F.softmax(pred["scar"], dim=1)
        pred_scar = pred_scar_proba.argmax(dim=1)
        scar_preds.extend(pred_scar)
        scar_probs.extend(pred_scar_proba[:, 1].double().tolist())

        # lvef prediction
        pred_lvef_proba = F.softmax(pred["lvef"], dim=1)
        pred_lvef = pred_lvef_proba.argmax(dim=1)
        lvef_preds.extend(pred_lvef)
        lvef_probs.extend(pred_lvef_proba[:, 1].double().tolist())

    if device == "cuda":
        result_df = pd.DataFrame(
            {
                "scar_true": scar_true,
                "lvef_true": lvef_true,
                "scar_pred": convert_tensor_list(scar_preds),
                "lvef_pred": convert_tensor_list(lvef_preds),
                "scar_proba": scar_probs,
                "lvef_proba": lvef_probs,
                "path": paths,
            }
        )
    else:
        result_df = pd.DataFrame(
            {
                "scar_true": scar_true,
                "lvef_true": lvef_true,
                "scar_pred": scar_preds,
                "lvef_pred": lvef_preds,
                "scar_proba": scar_probs,
                "lvef_proba": lvef_probs,
                "path": paths,
            }
        )
    return result_df


@torch.no_grad()
def predict_from_df_clinical(
    my_model,
    dataframe,
    input_transforms,
    image_size=(224, 224),
    lvef_threshold=50,
    device="cuda",
):
    """
    Given my_model and a dataframe, predict output class and probability
    and corresponding labels

    This inferencing paradigm is batch independent method.

    The function does not include model.eval(). Thus, the user must set model to eval mode before parsing this function.
    """

    def convert_tensor_list(tensor_list: list):
        return [t.cpu().item() for t in tensor_list]

    scar_true = dataframe.scar_cad.values.tolist()
    lvef_true = dataframe.LVEF.values.tolist()
    lvef_true = [1 if x <= lvef_threshold else 0 for x in lvef_true]
    paths = dataframe.path.values.tolist()
    # labels = dataframe.label.values.tolist()

    scar_preds = []
    lvef_preds = []
    scar_probs = []
    lvef_probs = []
    for i, row in tqdm(dataframe.iterrows()):
        img = Image.open(row["path"]).convert("RGB")
        img = input_transforms(image=np.array(img))["image"]
        clinical_cat = row[["female_gender", "dm", "ht", "smoke", "dlp"]]
        clinical_num = row[["age"]]

        features_num = clinical_num.to_numpy().astype(int)
        features_cat = clinical_cat.to_numpy().astype(float)

        inp = (
            img.view(1, 3, image_size[0], image_size[1]).to(device),
            torch.tensor(features_num, dtype=torch.float32).to(device).view(1, 1, -1),
            torch.tensor(features_cat, dtype=torch.long).to(device).view(1, 1, -1),
        )

        # print(inp_shape)
        pred = my_model(inp)

        # scar prediction
        pred_scar_proba = F.softmax(pred["scar"], dim=1)
        pred_scar = pred_scar_proba.argmax(dim=1)
        scar_preds.extend(pred_scar)
        scar_probs.extend(pred_scar_proba[:, 1].double().tolist())

        # lvef prediction
        pred_lvef_proba = F.softmax(pred["lvef"], dim=1)
        pred_lvef = pred_lvef_proba.argmax(dim=1)
        lvef_preds.extend(pred_lvef)
        lvef_probs.extend(pred_lvef_proba[:, 1].double().tolist())

    if device == "cuda":
        result_df = pd.DataFrame(
            {
                "scar_true": scar_true,
                "lvef_true": lvef_true,
                "scar_pred": convert_tensor_list(scar_preds),
                "lvef_pred": convert_tensor_list(lvef_preds),
                "negative_pred": [0 for i in range(len(scar_true))],
                "scar_proba": scar_probs,
                "lvef_proba": lvef_probs,
                "path": paths,
            }
        )
    else:
        result_df = pd.DataFrame(
            {
                "scar_true": scar_true,
                "lvef_true": lvef_true,
                "scar_pred": convert_tensor_list(scar_preds),
                "lvef_pred": convert_tensor_list(lvef_preds),
                "negative_pred": [0 for i in range(len(scar_true))],
                "scar_proba": scar_probs,
                "lvef_proba": lvef_probs,
                "path": paths,
            }
        )
    return result_df
