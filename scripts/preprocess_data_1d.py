import os
import sys
import os.path as op
import numpy as np
from glob import glob
import pandas as pd
from argparse import ArgumentParser
import pickle
from typing import List, Tuple
from tqdm.auto import tqdm
from PIL import Image, ImageOps, ImageDraw

module_dir = op.join(op.dirname(op.abspath(__file__)), "..")
sys.path.append(module_dir)
from mtecg import *
from scripts.preprocess_data import fix_duplicated_save_paths

## Position of ECG Leads
RAW_NUMBER_POSIX_OLD_FORMAT = {
    "I": [(29, 56), (8, 26)],
    "II": [(37, 275), (12, 249)],
    "III": [(50, 500), (14, 468)],
    "aVR": [(98, 721), (9, 686)],
    "aVL": [(92, 942), (7, 901)],
    "aVF": [(87, 1163), (10, 1122)],
    "V1": [(1539, 58), (1480, 21)],
    "V2": [(1536, 279), (1481, 247)],
    "V3": [(1532, 496), (1483, 469)],
    "V4": [(1536, 717), (1480, 679)],
    "V5": [(1534, 941), (1482, 908)],
    "V6": [(1533, 1163), (1483, 1125)],
}

RAW_NUMBER_POSIX_NEW_FORMAT = {
    "I": [(33, 77), (13, 46)],
    "II": [(63, 431), (13, 400)],
    "III": [(93, 785), (13, 754)],
    "aVR": [(858, 77), (770, 46)],
    "aVL": [(858, 431), (770, 400)],
    "aVF": [(858, 785), (770, 754)],
    "V1": [(1582, 77), (1525, 46)],
    "V2": [(1582, 431), (1525, 400)],
    "V3": [(1582, 785), (1525, 754)],
    "V4": [(2341, 77), (2285, 46)],
    "V5": [(2341, 431), (2285, 400)],
    "V6": [(2341, 785), (2285, 754)],
}

ECG_FORMAT_TO_LEAD_X_SIZE = {"old": 1472, "new": 701}


def remove_numbers(img: Image, posix_dict: dict):

    """
    mask numbers of leads with white rectangles
    """
    img_draw = ImageDraw.Draw(img)
    for k, shape in posix_dict.items():
        img_draw.rectangle(shape, fill="#FFFFFF", outline="#FFFFFF")

    return img


def list_of_all_leads(img_no_grid: Image.Image, ecg_format: str = "new"):
    """
    get all leads
    """
    if ecg_format == "old":
        row_length = 240
        row_starts = [0, 240, 480, 720, 960, 1080]
        row_ends = [num + row_length for num in row_starts]
        col_length = ECG_FORMAT_TO_LEAD_X_SIZE[ecg_format]
        col_starts = [0, 1472]
        col_ends = [num + col_length for num in col_starts]

    elif ecg_format == "new":
        row_length = 350
        row_starts = [0, 350, 700]
        row_ends = [num + row_length for num in row_starts]
        col_length = ECG_FORMAT_TO_LEAD_X_SIZE[ecg_format]
        col_starts = [0, 740, 1478, 2217]
        col_ends = [num + col_length for num in col_starts]

    leads = []
    for col_start, col_end in zip(col_starts, col_ends):
        for row_start, row_end in zip(row_starts, row_ends):
            leads.append(img_no_grid.crop((col_start, row_start, col_end, row_end)))
    return leads


def get_signal(image: Image):
    """
    get signal from cleaned single lead ecg image
    """

    image = ImageOps.grayscale(image)
    image_array = np.array(image)

    cy, cx = np.where((image_array[:, :] >= 0) & (image_array[:, :] <= 2))

    array_height = image_array.shape[0]
    cy = array_height - cy

    cx, cy = zip(*sorted(zip(cx, cy)))
    return np.array([cx, cy])


def extract_ecg_signal(
    path: str,
    ecg_format: str = "new",
    dpi: int = 300,
) -> Tuple[List[np.ndarray], List[Image.Image]]:
    """
    get list of ecg signal from path
    """
    if ecg_format == "old":
        box = (176, 258, 3126, 1672)
        posix_dict = RAW_NUMBER_POSIX_OLD_FORMAT

    elif ecg_format == "new":
        box = (82, 950, 3000, 2000)
        posix_dict = RAW_NUMBER_POSIX_NEW_FORMAT

    img = read_pdf_to_image(path, dpi=dpi, box=box)
    img = remove_grid_robust(np.array(img), n_jitter=3)
    img = remove_numbers(img, posix_dict)

    lead_image_list = list_of_all_leads(img, ecg_format=ecg_format)
    lead_array_coordinates_list = [get_signal(lead_image) for lead_image in lead_image_list]
    return lead_array_coordinates_list, lead_image_list


def get_path(
    row,
    parent_dir="./siriraj_data/ECG_MRI",
):
    """Get the full path to the source pdf file"""
    if "train" in row["split"] or row["split"] == "old_valid" or row["split"] == "old_test":
        row["path"] = op.join(
            parent_dir,
            "ECG_80_Training_dataset",
            str(row["Year"]),
            str(row["Month"]),
            row["File_Name"] + ".pdf",
        )
    elif "valid" in row["split"]:
        row["path"] = op.join(
            parent_dir,
            "ECG_10_Development_dataset",
            str(row["Year"]),
            str(row["Month"]),
            row["File_Name"] + ".pdf",
        )

    elif row["split"] == "new_test":
        row["path"] = op.join(
            "../datasets/siriraj_data/ECG_MRI_10%_test",
            str(row["Year"]),
            str(row["Month"]),
            row["File_Name"] + ".pdf",
        )
    return row


def fill_gaps(x_coordinate_array, y_coordinate_array, ecg_format: str = "new"):
    # x should be 0, 1, 2, ..., max(x)
    # y should be interpolated values for the missing x values

    temp_df = (
        pd.DataFrame({"x": x_coordinate_array, "y": y_coordinate_array}).groupby("x").mean().reset_index().astype(int)
    )
    temp_df = (
        temp_df.set_index("x")
        .reindex(range(0, ECG_FORMAT_TO_LEAD_X_SIZE[ecg_format] + 1))
        .reset_index()
        .interpolate()
        .fillna(method="bfill")
        .astype(int)
    )
    x_array, y_array = zip(*temp_df.values)
    return x_array, y_array


def preprocess(row):
    pdf_path = row["path"]
    save_path = row["save_path"]
    ecg_format = row["ecg_format"]

    try:
        lead_array_coordinates_list, lead_image_list = extract_ecg_signal(
            pdf_path,
            ecg_format=ecg_format,
        )
        x_coordinate_array_list = [lead_coordinates[0] for lead_coordinates in lead_array_coordinates_list]
        y_coordinate_array_list = [lead_coordinates[1] for lead_coordinates in lead_array_coordinates_list]

        clean_x_coordinates, clean_y_coordinates = [], []
        for x_array, y_array in zip(x_coordinate_array_list, y_coordinate_array_list):
            x_array, y_array = fill_gaps(x_array, y_array, ecg_format=ecg_format)
            # plt.plot(x_array, y_array)
            clean_x_coordinates.append(x_array)
            clean_y_coordinates.append(y_array)

        assert len(clean_x_coordinates) == 12
        assert (
            np.unique([len(x) for x in clean_x_coordinates]).shape[0] == 1
        ), f"Different lengths of x coordinates {[len(x) for x in clean_x_coordinates]}"
        for i, (x, y) in enumerate(zip(clean_x_coordinates, clean_y_coordinates)):
            assert len(x) == len(y), f"Length of x and y do not match for lead {i}"
            assert len(x) == len(set(x)), f"Duplicate x values for lead {i}"

        with open(save_path, "wb") as f:
            pickle.dump(clean_y_coordinates, f)

        return True
    except Exception as e:
        print(f"Error in {pdf_path}")
        print(e)
        return pdf_path


def main(args):
    if not args.save_dir:
        raise ValueError("Please specify the save directory")
    os.makedirs(args.save_dir, exist_ok=True)

    label_df = pd.read_csv(args.label_csv_path)
    label_df = label_df.apply(lambda row: get_path(row, parent_dir=args.parent_dir), axis=1)
    label_df = label_df[label_df["path"] != 0].reset_index(drop=True)
    label_df["is_grid"] = label_df.apply(lambda row: 1 if row["Year"] >= 2017 else 0, axis=1)

    label_df["save_path"] = label_df.apply(
        lambda row: op.join(args.save_dir, f"{row['File_Name']}_{row['run_num']}"), axis=1
    )
    # Replace \\ with / to avoid path issues.
    label_df["path"] = label_df["path"].apply(lambda x: x.replace("\\", "/"))
    label_df["save_path"] = label_df["save_path"].apply(lambda x: x.replace("\\", "/"))
    ## fix duplicated file names for saving purposes (modify the save_path in label_df accordingly)
    label_df = fix_duplicated_save_paths(label_df, pdf_parent_dir=args.parent_dir, save_extension="")

    label_df["ecg_format"] = label_df["is_grid"].apply(lambda has_grid: "new" if has_grid else "old")

    label_df["file_exists"] = label_df["save_path"].apply(lambda x: op.exists(x))
    to_process_df = label_df[label_df["file_exists"] == False].reset_index(drop=True).copy()

    print(f"Processing {len(to_process_df)} files")
    for _, row in tqdm(to_process_df.iterrows()):
        result = preprocess(
            row,
        )
        # results += [result]

    label_df.drop(columns=["is_grid", "file_exists"], inplace=True)
    label_df.to_csv(
        args.label_save_path,
        index=False,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--parent_dir", type=str, default="../datasets/siriraj_data/ECG_MRI")
    parser.add_argument("--save_dir", type=str, default="../datasets/siriraj_data/ECG_MRI_1d_data_interp")
    parser.add_argument("--label_csv_path", type=str, default="../datasets/all_ECG_cleared_duplicate_may23_final.csv")
    parser.add_argument(
        "--label_save_path", type=str, default="../datasets/all_ECG_cleared_duplicate_may23_final_1d_labels.csv"
    )

    args = parser.parse_args()
    main(args)
