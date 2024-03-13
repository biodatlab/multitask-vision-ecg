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
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

tqdm.pandas()

import biosppy
import pickle
import wandb
import torch
from torch.utils.data import DataLoader

module_dir = op.join(op.dirname(op.abspath(__file__)), "..")
sys.path.append(module_dir)
from mtecg import (
    ECG1DDataset,
    ECGClinical1DDataset,
    MultiTask1DDataset,
    MultiTaskClinical1DDataset,
    SingleTaskModel1D,
    SingleTaskClinicalModel1D,
    MultiTaskModel1D,
    MultiTaskClinicalModel1D,
)
from mtecg.utils import load_ecg_dataframe_1d, find_best_thresholds, apply_thresholds
import mtecg.constants as constants

MODEL_TYPE_TO_CLASS_MAPPING = {
    "single-task-1d": SingleTaskModel1D,
    "single-task-clinical-1d": SingleTaskClinicalModel1D,
    "multi-task-1d": MultiTaskModel1D,
    "multi-task-clinical-1d": MultiTaskClinicalModel1D,
}
clinical_feature_columns = ["age", "female_gender", "dm", "ht", "smoke", "dlp"]

ECG_FORMAT_TO_PAPER_PIXEL_SAMPLE_RATE = {"old": 294, "new": 280}


def apply_rpeaks_normalization(lead_array_list, rpeak_index, ecg_format):
    paper_pixel_sample_rate = ECG_FORMAT_TO_PAPER_PIXEL_SAMPLE_RATE[ecg_format]
    start = max(0, int(rpeak_index - 0.5 * paper_pixel_sample_rate))
    end = int(rpeak_index + 1.5 * paper_pixel_sample_rate)
    return [lead[start:end] for lead in lead_array_list]


def has_empty_leads(lead_array_list):
    for lead in lead_array_list:
        if len(lead) == 0:
            return True
    return False


def filter_out_empty_leads(dataframe):
    print(f"Shape Before filter out: {dataframe.shape}")
    dataframe["has_empty_leads"] = dataframe["lead_arrays"].progress_apply(lambda lead_list: has_empty_leads(lead_list))
    dataframe = dataframe[dataframe["has_empty_leads"] != True].reset_index(drop=True)
    print(f"Shape After filter out: {dataframe.shape}")
    return dataframe


def process_dataframe(dataframe: pd.DataFrame, configs: dict, scaler: StandardScaler = None):
    sample_rate = configs["sample_rate"]
    scaler = StandardScaler() if scaler is None else scaler
    dataframe["lead_arrays"] = dataframe[constants.path_column_name].progress_apply(
        lambda x: pickle.load(open(x, "rb"))
    )
    dataframe["lead_arrays"] = dataframe.progress_apply(
        lambda row: apply_rpeaks_normalization(row["lead_arrays"], row["median_first_rpeak_index"], row["ecg_format"]),
        axis=1,
    )
    dataframe = filter_out_empty_leads(dataframe)
    dataframe["lead_arrays"] = dataframe["lead_arrays"].progress_apply(
        lambda lead_list: [resample(lead, sample_rate) for lead in lead_list]
    )
    scaler.fit(np.array(dataframe["lead_arrays"].tolist()).reshape(-1, 12))

    dataframe["lead_arrays"] = dataframe["lead_arrays"].progress_apply(
        lambda lead_list: scaler.transform(np.array(lead_list).reshape(-1, 12))
    )
    return dataframe, scaler


def init_dataset(dataframe: pd.DataFrame, configs: dict):
    model_type = configs["model_type"]
    task = configs.get("task", "")
    lvef_threshold = configs.get("lvef_threshold", None)

    dataset_kwargs = {
        "dataframe": dataframe,
    }
    # if lvef_threshold is not None:
    #     dataset_kwargs["lvef_threshold"] = lvef_threshold

    if "single" in model_type:
        if task == "scar":
            if "clinical" in model_type:
                return ECGClinical1DDataset(**dataset_kwargs, label_column=constants.scar_label_column_name)
            return ECG1DDataset(**dataset_kwargs, label_column=constants.scar_label_column_name)
        elif task == "lvef":
            if "clinical" in model_type:
                return ECGClinical1DDataset(**dataset_kwargs, label_column=constants.lvef_label_column_name)
            return ECG1DDataset(**dataset_kwargs, label_column=constants.lvef_label_column_name)
        else:
            raise ValueError(f"task {task} is not supported in single task model.")

    elif "multi" in model_type:
        if "clinical" in model_type:
            return MultiTaskClinical1DDataset(**dataset_kwargs)
        return MultiTask1DDataset(**dataset_kwargs)


def get_dataloaders(data_dir: str, csv_path: str, configs: dict):
    dataframe = load_ecg_dataframe_1d(csv_path, data_dir)
    print(dataframe.shape)

    save_dir = op.join(configs["parent_save_dir"], get_run_name(configs))
    os.makedirs(save_dir, exist_ok=True)

    # Combine old train and new train.
    train_df = dataframe[dataframe.split.isin(["old_train", "new_train"])].reset_index()
    # Combine old valid and new valid.
    valid_df = dataframe[dataframe.split.isin(["old_valid", "new_valid"])].reset_index()

    # Get the old train and valid if scheme is 'pretrain'.
    if configs["dataset"] == "pretrain":
        train_df = dataframe[dataframe.split.isin(["old_train"])].reset_index()
        valid_df = dataframe[dataframe.split.isin(["old_valid"])].reset_index()
    # Get the new train and valid if scheme is 'transfer'.
    elif configs["dataset"] == "transfer":
        train_df = dataframe[dataframe.split.isin(["new_train"])].reset_index()
        valid_df = dataframe[dataframe.split.isin(["new_valid"])].reset_index()
    elif configs["dataset"] == "all":
        train_df = dataframe[dataframe.split.isin(["old_train", "new_train"])].reset_index()
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

    train_df, train_scaler = process_dataframe(train_df, configs)
    valid_df, _ = process_dataframe(valid_df, configs, scaler=train_scaler)

    scaler_save_path = op.join(configs["parent_save_dir"], get_run_name(configs), "scaler.joblib")
    joblib.dump(train_scaler, scaler_save_path)

    # Init datasets.
    train_dataset = init_dataset(train_df, configs)
    valid_dataset = init_dataset(valid_df, configs)

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
    run_suffix = f"{configs['sample_rate']}"
    if "lvef_threshold" in configs.keys():
        run_suffix += f"_LVEF{str(configs['lvef_threshold'])}"

    if "clinical" in configs["model_type"]:
        run_suffix += f"_{configs['rnn_type']}_dim{configs['rnn_output_size']}"

    run_name = f"{configs['backbone']}_{run_suffix}"
    return run_name


def main(args):
    SEED = 42
    np.random.seed(SEED)
    seed_everything(SEED, workers=True)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(SEED)

    # Load configs.
    configs = json.load(open(args.config_path))
    # Create parent save dir.
    parent_save_dir = configs["parent_save_dir"]
    os.makedirs(parent_save_dir, exist_ok=True)
    # Create run name.
    run_name = get_run_name(configs)

    # Init dataloaders.
    train_loader, valid_loader = get_dataloaders(args.data_dir, args.csv_path, configs=configs)

    # Init model.
    model = get_model(configs)
    if configs["dataset"] == "transfer":
        # Explicitly specify train=True to instantiate the model with loss functions on the correct device.
        model = model.from_configs(configs["pretrained_model_path"], train=True, device=configs["device"])

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../datasets/siriraj_data/ECG_MRI_1d_data_interp/")
    parser.add_argument(
        "--csv_path", type=str, default="../datasets/all_ECG_cleared_duplicate_may23_final_1d_labels_with_rpeaks.csv"
    )
    parser.add_argument("--config_path", type=str, default="configs-1d/single-task-scar.json")

    parser.add_argument("--project_name", type=str, default="mtecg")

    args = parser.parse_args()
    main(args)
