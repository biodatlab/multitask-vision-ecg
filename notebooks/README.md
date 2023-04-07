# Notebooks


This directory contains a collection of Jupyter notebooks that demonstrate the data preparation and analysis workflow for the project.

## Population Statistics

- [`get_data_statistics.ipynb`](get_data_statistics.ipynb) - This notebook demonstrates how to get the prevalence of each clinical feature in the dataset. The prevalence of each type of myocardial scarring and the prevalence of LVEF<40% and LVEF<50% are also calculated.

## Training


- [`ecg-single-task.ipynb`](ecg-single-task.ipynb) - This notebook demonstrates how to train a single-task model on the ECG data. The model is trained on both old and new ECG formats. The resulting model can be used to predict **either** the presence of myocardial scarring **or** the LVEF range. 

- [`ecg-multi-task.ipynb`](ecg-multi-task.ipynb) - This notebook demonstrates how to train a multi-task model on the ECG data. The model is trained on both old and new ECG formats. The resulting model can be used to predict **both** the presence of myocardial scarring **and** the LVEF range. *This model is the best performing model thus far when clinical features are absent.*

- [`ecg-multi-task-transfer.ipynb`](ecg-multi-task-transfer.ipynb) - This notebook demonstrates how to train a multi-task model on the ECG data using transfer learning. The model is first pretrained on old-format ECGs and is then finetuned on the new-format ECGs. The resulting model can be used to predict **both** the presence of myocardial scarring **and** the LVEF range.

- [`ecg-multi-task-with-clinical-features.ipynb`](ecg-multi-task-with-clinical-features.ipynb) - This notebook demonstrates how to train a multi-task model on the ECG data. The model is trained on both old and new **ECG data and their clinical features**. The resulting model can be used to predict **both** the presence of myocardial scarring **and** the LVEF range. *This model is the best performing model thus far when clinical features are included.*
- [`xgb-with-clinical-features.ipynb`](xgb-with-clinical-features.ipynb) - This notebook demonstrates how to train single-task XGBoost models on clinical features. The model is trained on **clinical features** of both old and new data. The resulting model can be used to predict **either** the presence of myocardial scarring **or** the LVEF range. *This model is the baseline model.*

## **Evaluation**

- [`evaluation.ipynb`](evaluation.ipynb) - This notebook demonstrates how to evaluate the performance of the models on the test set. The predicted probability from each model is generated here for further analysis.
- [`get_figures.ipynb`](get_figures.ipynb) - This notebook demonstrates how to generate the figures in the paper. This notebook uses the predicted probability generated from the evaluation notebook.
