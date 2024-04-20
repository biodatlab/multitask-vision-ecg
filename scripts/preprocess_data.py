import os
import sys
import os.path as op
import numpy as np
from glob import glob
import pandas as pd
from collections import Counter
import time
from argparse import ArgumentParser

module_dir = op.join(op.dirname(op.abspath(__file__)), "..")
sys.path.append(module_dir)
from mtecg import *


def get_path(
    row,
    parent_dir="./siriraj_data/ECG_MRI",
):
    """Get the full path to the source pdf file"""
    if row["train_80_percent"] == 1:
        row["path"] = op.join(
            parent_dir,
            "ECG_80_Training_dataset",
            str(row["year"]),
            str(row["Month"]),
            row["File_Name"] + ".pdf",
        )
    else:
        row["path"] = op.join(
            parent_dir,
            "ECG_10_Development_dataset",
            str(row["year"]),
            str(row["Month"]),
            row["File_Name"] + ".pdf",
        )
    return row


def is_grid(row):
    row["is_grid"] = 1 if row["year"] >= 2017 else 0
    return row


def get_save_path(
    row, save_dir: str, save_extension: str = ".jpg"
):
    row["save_path"] = op.join(save_dir, row["File_Name"] + save_extension)
    return row


def fix_duplicated_save_paths(
    label_dataframe,
    pdf_parent_dir: str,
    save_extension: str = ".jpg",
):
    all_paths = glob(f"{pdf_parent_dir}/*/*/*/*.pdf")
    ## find duplicate filenames
    count_dict = dict(Counter([Path(p).stem for p in all_paths]))
    duplicate_filenames = [
        k for k, v in count_dict.items() if v > 1
    ]
    ## select only duplicate files that have been labelled
    duplicate_df = label_dataframe[
        label_dataframe["File_Name"].isin(duplicate_filenames)
    ]
    unique_names = duplicate_df.File_Name.unique()

    for unique_name in tqdm(unique_names):
        subset_df = duplicate_df[duplicate_df["File_Name"] == unique_name].reset_index()
        for i, row in subset_df.iterrows():
            if i > 0:
                ## rename the file to be unique
                row["save_path"] = row["save_path"].replace(
                    save_extension, f"_{i}{save_extension}"
                )
                ## update the name in the original df
                label_dataframe.at[i, "save_path"] = row["save_path"]
    return label_dataframe


def preprocess(row, thread_count=8):
    image_path = row["path"]
    save_path = row["save_path"]
    try:
        ## if the image has grids, we need to remove them and rearrange the image
        if row["is_grid"]:
            image = read_pdf_to_image(image_path, box=(82, 950, 3000, 2000))
            image_no_grid = remove_grid_robust(np.array(image), n_jitter=3, thread_count=thread_count)
            image_rearranged = rearrange_leads(image_no_grid)
            image_rearranged.save(save_path)
        else:
            image = read_pdf_to_image(image_path)
            image.save(save_path)
        return True
    except:
        return image_path

def main(args):
    if not args.save_dir:
        raise ValueError("Please specify the save directory")
    os.makedirs(args.save_dir, exist_ok=True)

    label_df = pd.read_excel(args.label_excel_path, engine="openpyxl")
    label_df = label_df.apply(lambda row: get_path(row, parent_dir=args.parent_dir), axis=1)
    label_df = label_df.apply(is_grid, axis=1)
    label_df = label_df.apply(
        lambda row: get_save_path(row, save_dir=args.save_dir, save_extension=args.save_extension), axis=1
    )
    ## fix duplicated file names for saving purposes (modify the save_path in label_df accordingly)
    label_df = fix_duplicated_save_paths(
        label_df, pdf_parent_dir=args.parent_dir, save_extension=args.save_extension
    )

    if args.num_test_images:
        old_format_label_df = label_df[label_df["is_grid"] == False].reset_index(drop=True)
        new_format_label_df = label_df[label_df["is_grid"] == True].reset_index(drop=True)
        old_format_label_df = old_format_label_df.sample(args.num_test_images)
        new_format_label_df = new_format_label_df.sample(args.num_test_images)

        start = time.time()
        results = []
        for _, row in tqdm(old_format_label_df.iterrows()):
            result = preprocess(row, thread_count=args.thread_count)
            results += [result]
        end = time.time()
        print(f"Total time taken for Old-format: {end-start:.4f} seconds for {len(old_format_label_df)} images")

        start = time.time()
        results = []
        for _, row in tqdm(new_format_label_df.iterrows()):
            result = preprocess(row, thread_count=args.thread_count)
            results += [result]
        end = time.time()
        print(f"Total time taken for New-format: {end-start:.4f} seconds for {len(new_format_label_df)} images")

    else:
        start = time.time()
        results = []
        for _, row in tqdm(label_df.iterrows()):
            result = preprocess(row, thread_count=args.thread_count)
            results += [result]
        end = time.time()
        print(f"Total time taken: {end-start:.4f} seconds for {len(label_df)} images")
        ## save the new label file (of the images) to be used for training
        label_df.to_csv(args.label_save_path, index=False,)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--parent_dir", type=str, default="../datasets/siriraj_data/ECG_MRI")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--label_excel_path", type=str, default="../datasets/siriraj_data/ECG_MRI/ECG_MRI_80_training_10_development_for_aj_my_220707.1.xlsx")
    parser.add_argument("--label_save_path", type=str, default="../datasets/siriraj_data/ECG_MRI/ECG_MRI_80_training_10_development_220707_image_labels.xlsx")
    parser.add_argument("--save_extension", type=str, default=".jpg")
    parser.add_argument("--thread_count", type=int, default=8)

    parser.add_argument("--num_test_images", type=int, default=0, help="Number of images to use for testing the preprocessing pipeline")

    args = parser.parse_args()
    main(args)