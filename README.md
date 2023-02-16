# Vision-based Myocardial Scar and Left Ventricular Ejection Fraction Classification for 12-lead ECG Scans using Multi-task DNN

We propose a multi-task myocardial scar (MS) and left ventricular ejection fraction (LVEF) prediction using ECG scans by
developing deep learning models using over 14,000 ECG scans. Our model achieved an area under the receiver operating curve
(AUC) of 0.837 and 0.934 on MS and LVEF classification tasks, outperforming cardiologists. We verified the model against
a control population and got an accuracy of over 99 percent on both tasks. Our study demonstrates the potential of MS and
LVEF classification by ECGs in clinical screening.

## Repository structure

- `mtecg`: core library for multi-task MS and LVEF classification
- `notebooks`: Jupyter notebooks for experiments
- `app`: frontend and backend code for the screening application

## Multi-task ECG Classification

`mtecg` contains a core library for multi-task MS and LVEF classification. This include the preprocessing pipeline,
model architecture, and evaluation scripts.

## Experiment notebooks

`notebooks` contains Jupyter notebooks for experiments including:

- Multi-task MS and LVEF classification
- Transferred Multi-task MS and LVEF classification
- Single-task MS and LVEF classification
- Multi-task MS and LVEF classification with clinical data

## Screening application

`app` folder contains the frontend and backend code for the screening application (in Thai).
We design a UI for the application using [NEXT.js](https://nextjs.org/) and implement the backend
using [FastAPI](https://fastapi.tiangolo.com/).
