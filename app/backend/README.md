# Backend

This folder contains FastAPI source code for the backend of the application.

## Requirements

```sh
pip install -r requirements.txt

# for mac user, poppler need to be installed via conda
conda install -c conda-forge poppler
```

## FastAPI backend

Run the following script to start a FastAPI application.

```py
uvicorn api:app --reload
```
