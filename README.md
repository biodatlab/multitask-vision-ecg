# Computer vision-based myocardial scar and left ventricular ejection fraction classification for 12-lead electrocardiography scans using a multi-task deep neural network

- **Background:** Myocardial scar (MS) and left ventricular ejection fraction (LVEF) are important cardiovascular parameters. Cardiac magnetic resonance (CMR) can accurately access MS and LVEF, but it is costly and often unavailable in limited-resource settings. Electrocardiogram (ECG) is a widely used alternative due to its affordability and availability. The most commonly used ECG devices produce a printed image that is manually read. Alternatively, computer-based ECG interpretation could yield broad and far-reaching benefits. 
- **Methods:** Our study describes the development of a computer vision-based multi-task deep learning prediction model to read ECG scan images to distinguish MS and LVEF. Multiple deep learning models were developed on retrospective dataset of 14,052 ECGs with corresponding ground truth labels from CMR, which were separated into training, development, and test datasets. We also added clinical features to the model to compare the prediction performance. Our models were evaluated on the test dataset, prevalence-specific dataset, and against cardiologists.
- **Findings:** Our best model, the multi-task with clinical features, which yielded an area under the receiver operating curve of 0·848 (95%CI: 0·822 - 0·871) and 0·939 (95%CI: 0·921 - 0·954) for MS and LVEF classification, respectively, outperformed cardiologists. The AUC for MS classification of the multi-task both formats models was 0·835 (95%CI: 0·828 - 0·843) in the prevalence-specific test dataset.
Interpretation: Our results demonstrate the potential of computer-based MS and LVEF classification from ECG scan images in clinical screening. Our results demonstrate the potential of computer-based MS and LVEF classification from ECG scan images in clinical screening.

<img src="images/workflow.png"  width="700">


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
