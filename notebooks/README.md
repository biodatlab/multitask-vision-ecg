# Notebooks


This directory contains a collection of Jupyter notebooks that demonstrate the data preparation and analysis workflow for the project.

## Population Statistics

- [`get_dataset_statistics.ipynb`](get_dataset_statistics.ipynb) - This notebook demonstrates how to get the prevalence of each clinical feature in the dataset. The prevalence of each type of myocardial scarring and the prevalence of LVEF<40% and LVEF<50% are also calculated.

## Training

- [`xgb-with-clinical-features.ipynb`](xgb-with-clinical-features.ipynb) - This notebook demonstrates how to train single-task XGBoost models on clinical features. The model is trained on **clinical features** of both old and new data. The resulting model can be used to predict **either** the presence of myocardial scarring **or** the LVEF range. *This model is the baseline model.*

## **Evaluation**

- [`evaluation.ipynb`](evaluation.ipynb) - This notebook demonstrates how to evaluate the performance of the models on the test set. The predicted probability from each model is generated here for further analysis.
- [`evaluation_1d.ipynb`](evaluation_1d.ipynb) - This notebook demonstrates how to evaluate the performance of the models on the 1d test set. The predicted probability from each model is generated here for further analysis.

## **Figures**

- [`get_figures.ipynb`](get_figures.ipynb) - This notebook demonstrates how to generate the figures in the paper. This notebook uses the predicted probability generated from the evaluation notebook.
- [`get_figures_1d.ipynb`](get_figures_1d.ipynb) - This notebook demonstrates how to generate the figures in the paper for 1d experiments. This notebook uses the predicted probability generated from the evaluation notebook.