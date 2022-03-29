# Backend

This folder contains small a library for processing ECG files
and FastAPI for model prediction.

## Requirements

```sh
pip install -r requirements.txt

# for mac user, poppler need to be installed via conda
conda install -c conda-forge poppler
```

## ECG utilities

A package contains utility functions for prediction.

```py
from ecg_utils import load_learner_path, predict_path
learner = load_learner_path("models/model.pkl")
prediction = predict_path(learner, path) # predict from a given path
```

## FastAPI backend

Run the following script to start a FastAPI application.

```py
uvicorn api:app --reload
```
