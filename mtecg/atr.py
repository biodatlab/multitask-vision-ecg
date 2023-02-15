# Script for preprocessing various ECG dataset (we do not use these functions)
import os
import os.path as op
from typing import Optional
from glob import glob
import wfdb
import numpy as np
import biosppy
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.io import loadmat


def get_records(path: str = "mit-bih-arrhythmia-database-1.0.0/*.atr"):
    """
    Get ATR path from a given directory of MIT BIH dataet

    Paper: https://arxiv.org/pdf/1804.06812.pdf
    """
    paths = glob(path)
    paths = [p[:-4] for p in paths]
    paths.sort()
    return


def segmentation(path: str):
    """
    Segment EKG to a list of array of samples
    """
    normal = []
    signals, _ = wfdb.rdsamp(path, channels=[0])
    ann = wfdb.rdann(path, "atr")
    good = ["N"]
    ids = np.in1d(ann.symbol, good)
    imp_beats = ann.sample[ids]
    beats = ann.sample
    for i in imp_beats:
        beats = list(beats)
        j = beats.index(i)
        if (j != 0) and (j != len(beats) - 1):
            x = beats[j - 1]
            y = beats[j + 1]
            diff1 = abs(x - beats[j]) // 2
            diff2 = abs(y - beats[j]) // 2
            normal.append(signals[beats[j] - diff1 : beats[j] + diff2, 0])
    return normal


def get_ekg_code(header: list):
    """
    Get the EKG code from a given list of header.
    """
    dx = [r.split(":")[-1].strip() for r in header if "#dx" in r.lower()][0]
    return dx


def segment_ekg_pulses(data: np.array, peak_lead: int = 0, verbose: bool = False):
    """
    Given a 12 lead EKG data (n_leads, n_samples), find peaks and then
    segment it into each pulse

    Example usage:
    >>> data, header_data = load_ekg_data("data/E00001.mat")
    >>> ekg_pulses = segment_ekg_pulses(data)

    This function is also used in ``generate_ekg_images``
    """
    # use biosppy to detect peaks from the first lead
    peaks = biosppy.signals.ecg.christov_segmenter(signal=data[peak_lead, :], sampling_rate=200)[0]
    if verbose:
        print("Number of pulses = {}".format(len(peaks)))
    period = int((peaks[1:] - peaks[:-1]).mean())
    n = len(data[peak_lead, :])

    ekg_pulses = []
    for peak in peaks:
        ekg_pulses.append(data[:, max(peak - period // 2, 0) : min(peak + period // 2, n)])
    return ekg_pulses


def convert_ekg_to_image(
    signal: np.array,
    resize: tuple = (224, 224),
    filename: str = "ekg.png",
    linewidth: int = 3,
):
    """
    Convert a 1D signal in numpy array to an EKG image.

    filename provided is used as a temporary file to save the image.
    """
    fig = plt.figure(frameon=False)
    plt.plot(signal, color="k", linewidth=linewidth)
    plt.xticks([])
    plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    fig.savefig(filename)
    plt.close()
    img = Image.open(filename)
    img_gray = ImageOps.grayscale(img)
    if resize is not None:
        img_gray = img_gray.resize(resize)
    return img_gray


def generate_ekg_images(path: str, linewidth: int = 3, resize: Optional[tuple] = None, verbose: bool = False):
    """
    Generates EKG images from a given mat file path.
    Return a list of pulses where each one contains images of
    12 leads EKG.

    >>> ekg_images = generate_ekg_images("data/E00001.mat")
    """
    filename = Path(path).stem
    data, header_data = load_ekg_data(path)
    ekg_pulses = segment_ekg_pulses(data, verbose=verbose)  # segment EKG
    ekg_images = []
    for i_pulse, ekg_pulse in enumerate(ekg_pulses):
        temp_imgs = []
        for i in range(0, 12):
            img = convert_ekg_to_image(ekg_pulse[i, :], linewidth=linewidth, resize=resize)
            temp_imgs.append(img)
        ekg_images.append(
            {
                "images": temp_imgs,
                "filename": filename,
                "header": header_data,
                "path": path,
                "header_code": get_ekg_code(header_data),
                "pulse_number": i_pulse,
            }
        )
    return ekg_images


def load_ecg_matfile(filename: str):
    """
    Specify path to mat file, load a mat file and corresponding header file to
    data and header_data

    >>> data, header_data = load_ekg_data("data/E00001.mat")

    Reference:
        https://www.kaggle.com/bjoernjostein/georgia-12lead-ecg-challenge-database
    """
    x = loadmat(filename)
    data = np.asarray(x["val"], dtype=np.float64)
    new_file = filename.replace(".mat", ".hea")
    input_header_file = os.path.join(new_file)
    with open(input_header_file, "r") as f:
        header_data = f.readlines()
    return data, header_data
