import os
import sys
import json
import os.path as op
import numpy as np
from functools import partial
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger

import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

module_dir = op.join(op.dirname(op.abspath(__file__)), "..")
sys.path.append(module_dir)
from mtecg import (
    ScarDataset,
    LVEFDataset,
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


def get_valid_transforms():
    valid_transform = A.Compose([A.Resize(*image_size), A.Normalize(), ToTensorV2()])
    return valid_transform


def init_dataset(dataframe: pd.DataFrame, transforms: object, configs: dict):
    model_type = configs["model_type"]
    task = configs.get(["task"], "")
    lvef_threshold = configs.get(["lvef_threshold"], None)

    dataset_kwargs = {
        "dataframe": dataframe,
        "transformations": transforms,
    }
    if lvef_threshold is not None:
        dataset_kwargs["lvef_threshold"] = lvef_threshold

    if "single" in model_type:
        if task == "scar":
            return ScarDataset(**dataset_kwargs)
        elif task == "lvef":
            return LVEFDataset(**dataset_kwargs)
        else:
            raise ValueError(f"task {task} is not supported in single task model.")

    elif "multi" in model_type:
        if "clinical" in model_type:
            return MultiTaskClinicalCNNDataset(**dataset_kwargs)
        return MultiTaskDataset(**dataset_kwargs)


def get_dataloaders(image_dir: str, csv_path: str, configs: dict):
    dataframe = load_ecg_dataframe(csv_path, image_dir)

    # Combine old train and new train.
    train_df = dataframe[dataframe.split.isin(["old_train", "new_train"])].reset_index()
    # Combine old valid and new valid.
    valid_df = dataframe[dataframe.split.isin(["old_valid", "new_valid"])].reset_index()

    if "clinical" in configs["model_type"]:
        # Get imputer from train set.
        imputer = get_imputer(train_df)
        # Impute missing values in the train set.
        train_df[clinical_feature_columns] = imputer.transform(train_df[clinical_feature_columns])
        # Impute missing values in the valid set.
        valid_df[clinical_feature_columns] = imputer.transform(valid_df[clinical_feature_columns])

        # Find the best thresholds for imputing missing values from the train set.
        best_threshold_dict = find_best_thresholds(train_df)

        # Save the best thresholds.
        joblib.dump(
            best_threshold_dict,
            op.join(configs["parent_save_dir"], get_run_name["configs"], "imputer_threshold_dict.joblib"),
        )

        # Apply the best thresholds to the train set and the valid set.
        train_df = apply_thresholds(train_df, best_threshold_dict)
        valid_df = apply_thresholds(valid_df, best_threshold_dict)

    # Get train and valid transforms.
    train_transform = get_train_transforms()
    valid_transform = get_valid_transforms()

    # Init datasets.
    train_dataset = init_dataset(train_df, train_transform, configs)
    valid_dataset = init_dataset(valid_df, valid_transform, configs)

    # Init dataloaders.
    train_loader = DataLoader(train_dataset, batch_size=configs["batch_size"], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=configs["batch_size"], shuffle=False, pin_memory=True)

    return train_loader, valid_loader


def get_imputer(train_dataframe: pd.DataFrame):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    import joblib
    from sklearn.linear_model import LinearRegression

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
    run_suffix = f"{args.image_size}"
    if "lvef_threshold" in configs.keys():
        run_suffix += f"_LVEF{str(lvef_threshold)}"

    if "clinical" in configs["model_type"]:
        run_suffix += f"_{configs['rnn_type']}_dim{configs['rnn_output_size']}"

    run_name = f"{configs['backbone']}_{run_suffix}"
    return run_name


def main(args):
    # Load configs.
    configs = json.load(open(args.config_path))
    # Init dataloaders.
    train_loader, valid_loader = get_dataloaders(args.image_dir, args.csv_path, configs=configs)

    # Init model.
    model = get_model(configs)
    # Setup wandb logger.
    wandb_logger = setup_wandb_logger(project_name=args.project_name, configs=configs)
    logger.watch(
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

    # Init trainer.
    accumulate_grad_batches = configs.get("accumulate_grad_batches", 1)
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger,
        max_epochs=configs["num_epochs"],
        callbacks=[checkpoint_callback, StochasticWeightAveraging(1e-3)],
        accumulate_grad_batches=accumulate_grad_batches,
    )

    # Train model.
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    # Create run name.
    os.makedirs(configs["parent_save_dir"], exist_ok=True)
    run_name = get_run_name(configs)

    # Save model and configs.
    trainer.save_checkpoint(op.join(parent_save_dir, run_name, "model.ckpt"))
    model.save_configs(op.join(parent_save_dir, run_name))
    A.save(train_transform, op.join(parent_save_dir, run_name, "train_transform.json"))
    A.save(valid_transform, op.join(parent_save_dir, run_name, "transform.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="../../ecg/ecg-cnn-local/siriraj_data/ECG_MRI_images_new/")
    parser.add_argument("--csv_path", type=str, default="../../ECG_EF_Clin_train_dev_new.csv")
    parser.add_argument("--config_path", type=str, default="configs/multi-task.json")

    parser.add_argument("--project_name", type=str, default="mtecg")

    args = parser.parse_args()
    main(args)
