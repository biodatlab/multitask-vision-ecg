import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy

from mtecg.utils import categorize_lvef
import mtecg.constants as constants


class ScarDataset(Dataset):
    def __init__(self, dataframe, transformations):
        self.paths = list(dataframe[constants.path_column_name])
        self.labels = list(dataframe[constants.scar_label_column_name])
        self.transforms = transformations

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # dealing with the image
        image, scar_label = Image.open(self.paths[index]).convert("RGB"), self.labels[index]
        if self.transforms:
            image = self.transforms(image=np.array(image))["image"]
        return image, scar_label

class ScarClinicalDataset(ScarDataset):
    def __init__(self, dataframe, transformations):
        super(ScarClinicalDataset, self).__init__(dataframe, transformations)

        # Drop path, label, and numerical clinical features (age).
        self.categorical_feature_dataframe = dataframe[constants.categorical_feature_column_names]
        self.numerical_feature_dataframe = dataframe[[constants.age_column_name]]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        ### numerical clinical features should be separeted from categorical clinical features
        image, numerical_feature, categorical_features, label = (
            Image.open(self.paths[index]).convert("RGB"),
            self.numerical_feature_dataframe.iloc[[index]].to_numpy(),
            self.categorical_feature_dataframe.iloc[[index]].to_numpy(),
            self.labels[index],
        )
        if self.transforms:
            image = self.transforms(image=np.array(image))["image"]

        ### result format (x_image, x_numerical, x_categorical), y
        return (
            (
                image,
                torch.tensor(numerical_feature, dtype=torch.float32),
                torch.tensor(categorical_features, dtype=torch.long),
            ),
            label,
        )


class LVEFDataset(Dataset):
    def __init__(self, dataframe, transformations=None, lvef_threshold: int = 50):
        self.paths = list(dataframe[constants.path_column_name])
        self.labels = list(dataframe[constants.lvef_label_column_name])
        self.transforms = transformations
        self.lvef_threshold = lvef_threshold

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # dealing with the image
        image, label = Image.open(self.paths[index]).convert("RGB"), self.labels[index]
        if self.transforms:
            image = self.transforms(image=np.array(image))["image"]
        # label = categorize_lvef(label, threshold=self.lvef_threshold)
        return image, label

class LVEFClinicalDataset(LVEFDataset):
    def __init__(self, dataframe, transformations):
        super(LVEFClinicalDataset, self).__init__(dataframe, transformations, lvef_threshold)

        # Drop path, label, and numerical clinical features (age).
        self.categorical_feature_dataframe = dataframe[constants.categorical_feature_column_names]
        self.numerical_feature_dataframe = dataframe[[constants.age_column_name]]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        ### numerical clinical features should be separeted from categorical clinical features
        image, numerical_feature, categorical_features, label = (
            Image.open(self.paths[index]).convert("RGB"),
            self.numerical_feature_dataframe.iloc[[index]].to_numpy(),
            self.categorical_feature_dataframe.iloc[[index]].to_numpy(),
            self.labels[index],
        )
        if self.transforms:
            image = self.transforms(image=np.array(image))["image"]

        ### result format (x_image, x_numerical, x_categorical), y
        return (
            (
                image,
                torch.tensor(numerical_feature, dtype=torch.float32),
                torch.tensor(categorical_features, dtype=torch.long),
            ),
            label,
        )
class MultiTaskDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transformations=None, lvef_threshold: int = 50):
        self.paths = list(dataframe[constants.path_column_name])

        scar_labels = list(dataframe[constants.scar_label_column_name])
        lvef_labels = list(dataframe[constants.lvef_label_column_name])

        self.labels = list(zip(scar_labels, lvef_labels))
        self.lvef_threshold = lvef_threshold
        self.transforms = transformations

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # dealing with the image 
        image, label = Image.open(self.paths[index]).convert("RGB"), self.labels[index]
        if self.transforms:
            image = self.transforms(image=np.array(image))["image"]
        return image, {
            "scar": torch.tensor(label[0], dtype=torch.long),
            "lvef": torch.tensor(label[1], dtype=torch.long),
        }


class MultiTaskClinicalCNNDataset(MultiTaskDataset):
    def __init__(self, dataframe, transformations=None, lvef_threshold=50):
        super(MultiTaskClinicalCNNDataset, self).__init__(dataframe, transformations, lvef_threshold)

        # Drop path, label, and numerical clinical features (age).
        self.categorical_feature_dataframe = dataframe[constants.categorical_feature_column_names]
        self.numerical_feature_dataframe = dataframe[[constants.age_column_name]]

    def __getitem__(self, index):
        ### numerical clinical features should be separeted from categorical clinical features
        image, numerical_feature, categorical_features, label = (
            Image.open(self.paths[index]).convert("RGB"),
            self.numerical_feature_dataframe.iloc[[index]].to_numpy(),
            self.categorical_feature_dataframe.iloc[[index]].to_numpy(),
            self.labels[index],
        )
        if self.transforms:
            image = self.transforms(image=np.array(image))["image"]

        ### result format (x_image, x_numerical, x_categorical), y
        return (
            (
                image,
                torch.tensor(numerical_feature, dtype=torch.float32),
                torch.tensor(categorical_features, dtype=torch.long),
            ),
            {"scar": torch.tensor(label[0], dtype=torch.long), "lvef": torch.tensor(label[1], dtype=torch.long)},
        )
