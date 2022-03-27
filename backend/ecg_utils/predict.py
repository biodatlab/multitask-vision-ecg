import os.path as op
import pathlib
from pathlib import Path
import numpy as np
from fastbook import *
from .io import read_pdf_to_image


plt = platform.system()
if plt in ("Linux", "Darwin"):
    pathlib.WindowsPath = pathlib.PosixPath


def load_learner_path(model_path: str):
    """
    Load FastAI learner from a given path
    """
    assert op.exists(model_path), "Trained model does not exist."
    model_path = Path(model_path)
    learner = load_learner(model_path)  # load learner
    return learner


def predict_file(learner, path: str):
    """
    Predict ECG from a given path (can be PNG or PDF file).
    Return a class and probability of prediction.
    >>> pred, pred_proba = predict_ecg(path)  # path to PNG or PDF file
    """
    suffix = str(Path(path).suffix).lower()
    if suffix == ".png":
        prediction = learner.predict(path)
    elif suffix == ".pdf":
        img = read_pdf_to_image(path)
        prediction = learner.predict(np.array(img))
    else:
        print("Please provide a PNG or PDF file for classification.")
        return (None, None)
    pred, _, pred_proba = prediction
    return {"class": pred, "proba": [float(e) for e in list(pred_proba)]}


def predict_array(learner, image: np.array):
    """
    Predict ECG from a given image
    """
    prediction = learner.predict(image)
    pred, _, pred_proba = prediction
    return {"class": pred, "proba": [float(e) for e in list(pred_proba)]}
