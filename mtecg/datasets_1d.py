import numpy as np
import pandas as pd
from PIL import Image

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler

from mtecg.utils import categorize_lvef
import mtecg.constants as constants


class ECG1DDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, lead_arrays_column: str = "lead_arrays", label_column: str = "scar_cad"):
        self.lead_arrays_list = list(dataframe[lead_arrays_column])
        self.label_list = list(dataframe[label_column])

    def __len__(self):
        return len(self.lead_arrays_list)

    def __getitem__(self, index):
        lead_arrays = self.lead_arrays_list[index]
        label = self.label_list[index]
        return torch.tensor(lead_arrays, dtype=torch.float32), label


class ECGClinical1DDataset(ECG1DDataset):
    def __init__(self, dataframe, lead_arrays_column: str = "lead_arrays", label_column: str = "scar_cad"):
        super(ECGClinical1DDataset, self).__init__(dataframe, lead_arrays_column, label_column)

        # Drop path, label, and numerical clinical features (age).
        self.categorical_feature_dataframe = dataframe[constants.categorical_feature_column_names]
        self.numerical_feature_dataframe = dataframe[[constants.age_column_name]]

    def __getitem__(self, index):
        ### numerical clinical features should be separeted from categorical clinical features
        lead_arrays, numerical_feature, categorical_features, label = (
            self.lead_arrays_list[index],
            self.numerical_feature_dataframe.iloc[[index]].to_numpy(),
            self.categorical_feature_dataframe.iloc[[index]].to_numpy(),
            self.label_list[index],
        )

        ### result format (x_lead_arrays, x_numerical, x_categorical), y
        return (
            (
                torch.tensor(lead_arrays, dtype=torch.float32),
                torch.tensor(numerical_feature, dtype=torch.float32),
                torch.tensor(categorical_features, dtype=torch.long),
            ),
            label,
        )


class MultiTask1DDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, lead_arrays_column: str = "lead_arrays"):
        self.lead_arrays_list = list(dataframe[lead_arrays_column])

        scar_labels = list(dataframe[constants.scar_label_column_name])
        lvef_labels = list(dataframe[constants.lvef_label_column_name])

        self.label_list = list(zip(scar_labels, lvef_labels))

    def __len__(self):
        return len(self.lead_arrays_list)

    def __getitem__(self, index):
        # dealing with the image 
        lead_arrays, label = self.lead_arrays_list[index], self.label_list[index]
        return torch.tensor(lead_arrays, dtype=torch.float32), {
            "scar": torch.tensor(label[0], dtype=torch.long),
            "lvef": torch.tensor(label[1], dtype=torch.long),
        }


class MultiTaskClinical1DDataset(MultiTask1DDataset):
    def __init__(self, dataframe, lead_arrays_column: str = "lead_arrays"):
        super(MultiTaskClinical1DDataset, self).__init__(dataframe, lead_arrays_column)

        # Drop path, label, and numerical clinical features (age).
        self.categorical_feature_dataframe = dataframe[constants.categorical_feature_column_names]
        self.numerical_feature_dataframe = dataframe[[constants.age_column_name]]

    def __getitem__(self, index):
        ### numerical clinical features should be separeted from categorical clinical features
        lead_arrays, numerical_feature, categorical_features, label = (
            self.lead_arrays_list[index],
            self.numerical_feature_dataframe.iloc[[index]].to_numpy(),
            self.categorical_feature_dataframe.iloc[[index]].to_numpy(),
            self.label_list[index],
        )

        ### result format (x_lead_arrays, x_numerical, x_categorical), y
        return (
            (
                torch.tensor(lead_arrays, dtype=torch.float32),
                torch.tensor(numerical_feature, dtype=torch.float32),
                torch.tensor(categorical_features, dtype=torch.long),
            ),
            {"scar": torch.tensor(label[0], dtype=torch.long), "lvef": torch.tensor(label[1], dtype=torch.long)},
        )
