# Myocardial scar and left ventricular ejection fraction classification for electrocardiography image using multi-task deep learning

Published at [Scientific Reports](https://www.nature.com/articles/s41598-024-58131-6)

Myocardial scar (MS) and left ventricular ejection fraction (LVEF) are vital cardiovascular parameters, conventionally determined using cardiac magnetic resonance (CMR). However, given the high cost and limited availability of CMR in resource-constrained settings, electrocardiograms (ECGs) are a cost-effective alternative. We developed computer vision-based multi-task deep learning models to analyze 12-lead ECG 2D images, predicting MS and LVEF < 50%. Our dataset comprises 14,052 ECGs with clinical features, utilizing ground truth labels from CMR. Our top-performing model achieved AUC values of 0.838 (95% CI 0.812–0.862) for MS and 0.939 (95% CI 0.921–0.954) for LVEF < 50% classification, outperforming cardiologists. Moreover, MS predictions in a prevalence-specific test dataset recorded an AUC of 0.812 (95% CI 0.810–0.814). Extracted 1D signals from ECG images yielded inferior performance, compared to the 2D approach. In conclusion, our results demonstrate the potential of computer-based MS and LVEF < 50% classification from ECG scan images in clinical screening offering a cost-effective alternative to CMR.

<img src="images/workflow.png"  width="700">

We applied Grad-CAM++ to visualize the areas of ECG images that influenced the model decision. Figure below shows examples of heatmaps generated on top of the ECGs for multi-task and multi-task with clinical model.

<img src="images/gradcam.png"  width="700">

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
