import os
import sys
import json
import os.path as op
import numpy as np
import pandas as pd
from functools import partial
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import joblib
from sklearn.linear_model import LinearRegression

import wandb
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

module_dir = op.join(op.dirname(op.abspath(__file__)), "..")
sys.path.append(module_dir)
from mtecg import (
    ScarDataset,
    ScarClinicalDataset,
    LVEFDataset,
    LVEFClinicalDataset,
    MultiTaskDataset,
    MultiTaskClinicalCNNDataset,
    SingleTaskModel,
    SingleTaskClinicalCNNModel,
    MultiTaskModel,
    MultiTaskClinicalCNNModel,
)
from mtecg.utils import load_ecg_dataframe, find_best_thresholds, apply_thresholds


SEED = 42
np.random.seed(SEED)
seed_everything(SEED, workers=True)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(SEED)


MODEL_TYPE_TO_CLASS_MAPPING = {
    "single-task": SingleTaskModel,
    "single-task-clinical": SingleTaskClinicalCNNModel,
    "multi-task": MultiTaskModel,
    "multi-task-clinical": MultiTaskClinicalCNNModel,
}
clinical_feature_columns = ["age", "female_gender", "dm", "ht", "smoke", "dlp"]


def get_train_transforms(image_size: int = 384):
    image_size = (image_size, image_size)
    train_transform = A.Compose(
        [
            A.Resize(*image_size),
            A.Blur(blur_limit=3, p=0.2),
            A.RandomBrightnessContrast(),
            A.MotionBlur(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    return train_transform


def get_valid_transforms(image_size: int = 384):
    image_size = (image_size, image_size)
    valid_transform = A.Compose([A.Resize(*image_size), A.Normalize(), ToTensorV2()])
    return valid_transform


def init_dataset(dataframe: pd.DataFrame, transforms: object, configs: dict):
    model_type = configs["model_type"]
    task = configs.get("task", "")
    lvef_threshold = configs.get("lvef_threshold", None)

    dataset_kwargs = {
        "dataframe": dataframe,
        "transformations": transforms,
    }
    if lvef_threshold is not None:
        dataset_kwargs["lvef_threshold"] = lvef_threshold

    if "single" in model_type:
        if task == "scar":
            if "clinical" in model_type:
                return ScarClinicalDataset(**dataset_kwargs)
            return ScarDataset(**dataset_kwargs)
        elif task == "lvef":
            if "clinical" in model_type:
                return LVEFClinicalDataset(**dataset_kwargs)
            return LVEFDataset(**dataset_kwargs)
        else:
            raise ValueError(f"task {task} is not supported in single task model.")

    elif "multi" in model_type:
        if "clinical" in model_type:
            return MultiTaskClinicalCNNDataset(**dataset_kwargs)
        return MultiTaskDataset(**dataset_kwargs)


def get_dataloaders(image_dir: str, csv_path: str, configs: dict):
    dataframe = load_ecg_dataframe(csv_path, image_dir)

    save_dir = op.join(configs["parent_save_dir"], get_run_name(configs))
    os.makedirs(save_dir, exist_ok=True)

    # Combine old train and new train.
    train_df = dataframe[dataframe.split.isin(["old_train", "new_train"])].reset_index()
    # Combine old valid and new valid.
    valid_df = dataframe[dataframe.split.isin(["old_valid", "new_valid"])].reset_index()

    if "clinical" in configs["model_type"]:
        # Get imputer from train set.
        imputer = get_imputer(train_df, configs)
        # Impute missing values in the train set.
        train_df[clinical_feature_columns] = imputer.transform(train_df[clinical_feature_columns])
        # Impute missing values in the valid set.
        valid_df[clinical_feature_columns] = imputer.transform(valid_df[clinical_feature_columns])

        # Find the best thresholds for imputing missing values from the train set.
        best_threshold_dict = find_best_thresholds(train_df)

        # Save the best thresholds.
        joblib.dump(
            best_threshold_dict,
            op.join(save_dir, "imputer_threshold_dict.joblib"),
        )

        # Apply the best thresholds to the train set and the valid set.
        train_df = apply_thresholds(train_df, best_threshold_dict)
        valid_df = apply_thresholds(valid_df, best_threshold_dict)

    # Get train and valid transforms.
    train_transform = get_train_transforms(configs["image_size"])
    valid_transform = get_valid_transforms(configs["image_size"])

    # Init datasets.
    train_dataset = init_dataset(train_df, train_transform, configs)
    valid_dataset = init_dataset(valid_df, valid_transform, configs)

    # Init dataloaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=configs["num_workers"],
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=configs["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=configs["num_workers"],
    )

    return train_loader, valid_loader


def get_imputer(train_dataframe: pd.DataFrame, configs: dict):
    # Init imputer.
    imputer = IterativeImputer(missing_values=np.nan, max_iter=10, sample_posterior=True, random_state=42)

    # Fit the imputer on the train set.
    imputer.fit(train_dataframe[clinical_feature_columns])

    # Save the imputer.
    imputer_path = op.join(configs["parent_save_dir"], get_run_name(configs), "imputer.joblib")
    joblib.dump(imputer, imputer_path)

    return imputer


def get_model(configs: dict):
    model_type = configs["model_type"]
    model_class = MODEL_TYPE_TO_CLASS_MAPPING[model_type]
    model = model_class(**configs)
    return model


def setup_wandb_logger(project_name: str, configs: dict):
    run = wandb.init(project=project_name, save_code=True)
    run.log_code(".", include_fn=lambda path: path.endswith(".py"))
    run.config.update(
        {
            "batch_size": configs["batch_size"],
        }
    )

    wandb_logger = WandbLogger(
        name=configs["backbone"],
        project=project_name,
        # log_model=True,
    )
    return wandb_logger


def get_run_name(configs: dict):
    run_suffix = f"{configs['image_size']}"
    if "lvef_threshold" in configs.keys():
        run_suffix += f"_LVEF{str(configs['lvef_threshold'])}"

    if "clinical" in configs["model_type"]:
        run_suffix += f"_{configs['rnn_type']}_dim{configs['rnn_output_size']}"

    run_name = f"{configs['backbone']}_{run_suffix}"
    return run_name


def main(args):
    # Load configs.
    configs = json.load(open(args.config_path))
    # Create parent save dir.
    parent_save_dir = configs["parent_save_dir"]
    os.makedirs(parent_save_dir, exist_ok=True)
    # Create run name.
    run_name = get_run_name(configs)

    # Init dataloaders.
    train_loader, valid_loader = get_dataloaders(args.image_dir, args.csv_path, configs=configs)

    # Init model.
    model = get_model(configs)
    # Setup wandb logger.
    wandb_logger = setup_wandb_logger(project_name=args.project_name, configs=configs)
    wandb_logger.watch(
        model,
        # log_freq=300, # uncomment to log gradients
        log_graph=True,
    )

    # Init callbacks.
    checkpoint_callback = ModelCheckpoint(
        filename=configs["backbone"] + "{val_acc:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    earlystop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
    )

    # Init trainer.
    accumulate_grad_batches = configs.get("accumulate_grad_batches", 1)
    precision = configs.get("precision", "16-mixed")
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=wandb_logger,
        max_epochs=configs["num_epochs"],
        callbacks=[checkpoint_callback, earlystop_callback, StochasticWeightAveraging(1e-3)],
        accumulate_grad_batches=accumulate_grad_batches,
        precision=precision,
        log_every_n_steps=1,
    )

    # Train model.
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    # Save model and configs.
    trainer.save_checkpoint(op.join(parent_save_dir, run_name, "model.ckpt"))
    model.save_configs(op.join(parent_save_dir, run_name))
    A.save(get_train_transforms(configs["image_size"]), op.join(parent_save_dir, run_name, "train_transform.json"))
    A.save(get_valid_transforms(configs["image_size"]), op.join(parent_save_dir, run_name, "transform.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="../../ecg/ecg-cnn-local/siriraj_data/ECG_MRI_images_new/")
    parser.add_argument("--csv_path", type=str, default="../datasets/all_ECG_cleared_duplicate_may23_final.csv")
    parser.add_argument("--config_path", type=str, default="configs/multi-task.json")

    parser.add_argument("--project_name", type=str, default="mtecg")

    args = parser.parse_args()
    main(args)
