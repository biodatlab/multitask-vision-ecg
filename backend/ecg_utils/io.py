import numpy as np
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes


def read_pdf_to_image(path: str, dpi: int = 300, box: tuple = (180, 280, 3000, 1600)):
    """
    Convert an input PDF path to PIL image and crop with a given `box` parameter
    """
    img = convert_from_path(path, dpi=dpi)[0]  # assume reading first page
    img = img.crop(box=box)  # assume cropping with DPI=300
    return img


def read_bytes_to_image(
    pdf_bytes: bytes, dpi: int = 300, box: tuple = (180, 280, 3000, 1600)
):
    """
    Convert PDF to image locally. Accept bytes / byte-like objects of a pdf file.
    """
    img = convert_from_bytes(pdf_bytes, dpi=dpi)[0]
    img = img.crop(box=box)  # assume cropping with DPI=300
    return img


def remove_grid(img_crop: np.array):
    """
    Remove gridline from a given ECG scan
    """
    img_rm = np.array(img_crop)
    colors = [(216, 111, 98), (250, 200, 168), (168, 0, 0)]  # grid RGB values
    for c in colors:
        indices = np.where(
            (img_rm[:, :, 0] == c[0])
            & (img_rm[:, :, 1] == c[1])
            & (img_rm[:, :, 2] == c[2])
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
