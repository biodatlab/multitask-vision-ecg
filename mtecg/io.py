from typing import Optional, List, Tuple
import os
import os.path as op
from glob import glob
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product
import PIL
from PIL import Image
from pdf2image import convert_from_path
from sklearn.model_selection import train_test_split

OLD_FORMAT_IMAGE_SIZE = (2820, 1320)
OLD_FORMAT_ECG_BBOX = (180, 280, 3000, 1600)
NEW_FORMAT_ECG_BBOX = (82, 950, 3000, 2000)
RGB_VALUE_LIST = [(216, 111, 98), (250, 200, 168), (168, 0, 0)]
RGB_VALUE_VARIATION_RANGE = 2


def convert_pdf_to_image(path: str, dpi: int = 300, save_dir: Optional[str] = None):
    """
    Convert PDF to image to a save directory. Add page number to a saved file.
    """
    filename = Path(path).stem
    pages = convert_from_path(path, dpi=dpi)
    if save_dir is not None:
        if not op.exists(save_dir):
            os.makedirs(save_dir)
        for i, page in enumerate(pages):
            page.save(op.join(save_dir, f"{filename}_{i:02d}.png"), "PNG")
    return pages


def read_pdf_to_image(path: str, dpi: int = 300, box: tuple = (180, 280, 3000, 1600), thread_count=4):
    """
    Convert an input PDF path to PIL image and crop with a given `box` parameter
    """
    img = convert_from_path(path, dpi=dpi, thread_count=thread_count)[0]  # assume reading first page
    img = img.crop(box=box)  # assume cropping with DPI=300
    return img


# def remove_grid(img_crop: np.array):
#     """
#     Remove gridline from a given ECG scan
#     """
#     img_rm = np.array(img_crop)
#     colors = [(216, 111, 98), (250, 200, 168), (168, 0, 0)]  # grid RGB values
#     for c in colors:
#         indices = np.where(
#             (img_rm[:, :, 0] == c[0])
#             & (img_rm[:, :, 1] == c[1])
#             & (img_rm[:, :, 2] == c[2])
#         )
#         img_rm[indices] = np.array([255, 255, 255])
#     return Image.fromarray(img_rm)


# def generate_color_list(
#     colors: list = [(216, 111, 98), (250, 200, 168), (168, 0, 0)], n_jitter: int = 3
# ):
#     """
#     Generate pixel color around a given list of colors.
#     This method is used to generate wider colors for gridline removal.
#     """
#     # jitter range
#     jitter = list(range(-n_jitter, n_jitter))
#     jitter_array = np.vstack(
#         [np.array(j) for j in list(product(jitter, jitter, jitter))]
#     )

#     robust_colors = []
#     sel_colors = [np.array(color) for color in colors[:2]]
#     for color in sel_colors:
#         color_jitter = [tuple(c) for c in (color + jitter_array)]
#         robust_colors.extend(color_jitter)
#     robust_colors.extend(colors[2:])
#     return robust_colors


def remove_grid_robust(img_crop: np.array, n_jitter=RGB_VALUE_VARIATION_RANGE):
    """
    Remove grid lines from scanned ECG with more robustness.
    """

    img_rm = np.array(img_crop)
    red_channel_img = img_rm[:, :, 0]
    green_channel_img = img_rm[:, :, 1]
    blue_channel_img = img_rm[:, :, 2]

    # colors = generate_color_list(n_jitter=n_jitter)  # grid RGB values

    for (red, green, blue) in RGB_VALUE_LIST:
        indices = np.where(
            (red - n_jitter <= red_channel_img)
            & (red + n_jitter >= red_channel_img)
            & (green - n_jitter <= green_channel_img)
            & (green + n_jitter >= green_channel_img)
            & (blue - n_jitter <= blue_channel_img)
            & (blue + n_jitter >= blue_channel_img),
        )
        img_rm[indices] = np.array([255, 255, 255])
    return Image.fromarray(img_rm)


def rearrange_leads(img_no_grid: Image.Image):
    """
    Rearrange leads from a given grid-free ECG scan

    Usage
    =====
    >>> img = read_pdf_to_image("./siriraj_data/ecg_with_grid/30007923.pdf", box=(82, 950, 3000, 2000))
    >>> img_no_grid = remove_grid(np.array(img))
    >>> rearranged_ecg = rearrange_leads(img_no_grid)  # rearrange grid scan to non-grid scan format
    """
    row_length = 350
    row_starts = [0, 350, 700]
    row_ends = [num + row_length for num in row_starts]
    col_length = 701
    col_starts = [0, 740, 1478, 2217]
    col_ends = [num + col_length for num in col_starts]

    leads = []
    for col_start, col_end in zip(col_starts, col_ends):
        for row_start, row_end in zip(row_starts, row_ends):
            leads.append(img_no_grid.crop((col_start, row_start, col_end, row_end)))
    lead_stack_1 = np.vstack(leads[:6])
    lead_stack_2 = np.vstack(leads[6:])
    img_rearranged = np.hstack((lead_stack_1, lead_stack_2))
    return Image.fromarray(img_rearranged)


def prepare_df_from_folder(path: str, random_state: int = 3):
    """
    Prepare training, validation, and test dataframe from a given path
    to a folder.

    Folder has a structure of
        - scar/sub_folder/*.pdf
        - no_scar/sub_folder/*.pdf
    """
    scar_files = glob(op.join(path, "scar/*/*.pdf"))
    no_scar_files = glob(op.join(path, "no_scar/*/*.pdf"))
    label_df = pd.DataFrame(
        list(zip(scar_files, [1] * len(scar_files))) + list(zip(no_scar_files, [0] * len(no_scar_files))),
        columns=["path", "label"],
    )
    train_df, val_df = train_test_split(label_df, test_size=0.2, random_state=random_state)
    val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=random_state)
    return train_df, val_df, test_df


def create_data_folder(df: pd.DataFrame, folder="data/data_full_image/train/"):
    """
    Create a dataset folder from a given dataframe.

    >>> create_data_folder(train_df, folder="data/data_full_image/train/")
    >>> create_data_folder(val_df, folder="data/data_full_image/val/")
    >>> create_data_folder(test_df, folder="data/data_full_image/test/")
    """
    for p in ["Normal", "Abnormal"]:
        if not op.exists(op.join(folder, p)):
            os.makedirs(op.join(folder, p))
    for _, r in tqdm(df.iterrows()):
        img = read_pdf_to_image(r.path)
        w, h = img.size
        w, h = w // 3, h // 3
        img = img.resize((w, h))
        if r.Scar == "Normal" and not pd.isnull(r.Scar):
            img.save(op.join(folder, r.Scar, f"{r.File_name}.png"))
        elif r.Scar == "Abnormal" and not pd.isnull(r.Scar):
            img.save(op.join(folder, r.Scar, f"{r.File_name}.png"))
        else:
            pass


def partially_crop_old_format_ecg_image(
    image: PIL.Image.Image,
    first_column_bbox: tuple = (0, 0, 800, OLD_FORMAT_IMAGE_SIZE[1]),
    second_column_bbox: tuple = (1470, 0, 2200, OLD_FORMAT_IMAGE_SIZE[1]),
):
    first_column_image = image.crop(box=first_column_bbox)
    second_column_image = image.crop(box=second_column_bbox)
    stacked_image_array = np.hstack((np.array(first_column_image), np.array(second_column_image)))
    final_cropped_image = Image.fromarray(stacked_image_array)
    return final_cropped_image
