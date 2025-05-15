import logging
import os
import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image
from step1 import step1_python

logging.basicConfig(
    filename="/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/lung_mask/out.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

logging.basicConfig(
    filename="/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/lung_mask/err.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def process_mask(mask):
    """
    Process the given mask by performing convex hull operation and binary dilation.

    Parameters:
    mask (ndarray): The input mask to be processed.

    Returns:
    ndarray: The processed mask after convex hull and binary dilation operations.
    """
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 2 * np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask


def lumTrans(img):
    """
    Apply lung windowing transformation to the input image.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Transformed image.

    """
    lungwin = np.array([-1200.0, 600.0])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype("uint8")
    return newimg


def resample(imgs, spacing, new_spacing, order=2):
    """
    Resamples the input images to a new spacing.

    Args:
        imgs (ndarray): The input images to be resampled.
        spacing (ndarray): The current spacing of the images.
        new_spacing (ndarray): The desired spacing for the resampled images.
        order (int, optional): The interpolation order. Defaults to 2.

    Returns:
        ndarray: The resampled images.
        ndarray: The true spacing of the resampled images.
    """
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode="nearest", order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError("wrong shape")


def savenpy(id, filelist, prep_folder, data_path, use_existing=True):
    """
    Save the preprocessed lung mask and clean image for a given file.

    Args:
        id (int): The index of the file in the filelist.
        filelist (list): List of file names.
        prep_folder (str): Path to the folder where the preprocessed files will be saved.
        data_path (str): Path to the folder containing the original data files.
        use_existing (bool, optional): Whether to use existing preprocessed files if available. Defaults to True.
    """
    resolution = np.array([1, 1, 1])
    name = filelist[id]
    if use_existing:
        if os.path.exists(os.path.join(prep_folder, name + "_label.npy")) and os.path.exists(
            os.path.join(prep_folder, name + "_clean.npy")
        ):
            logging.info(name + " had been done")
            return
    try:
        im, m1, m2, spacing, nifti = step1_python(os.path.join(data_path, name))
        Mask = m1 + m2

        output_name = os.path.join(prep_folder, name + "_mask.npy")
        output_name = output_name.replace(".nii.gz", "")
        np.save(output_name, Mask)

        newshape = np.round(np.array(Mask.shape) * spacing / resolution)
        xx, yy, zz = np.where(Mask)
        box = np.array(
            [
                [np.min(xx), np.max(xx)],
                [np.min(yy), np.max(yy)],
                [np.min(zz), np.max(zz)],
            ]
        )
        box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
        box = np.floor(box).astype("int")
        margin = 5
        extendbox = np.vstack(
            [
                np.max([[0, 0, 0], box[:, 0] - margin], 0),
                np.min([newshape, box[:, 1] + 2 * margin], axis=0).T,
            ]
        ).T
        extendbox = extendbox.astype("int")

        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1 + dm2
        Mask = m1 + m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        im[np.isnan(im)] = -2000
        sliceim = lumTrans(im)
        sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype("uint8")
        bones = sliceim * extramask > bone_thresh
        sliceim[bones] = pad_value
        sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
        sliceim2 = sliceim1[
            extendbox[0, 0] : extendbox[0, 1],
            extendbox[1, 0] : extendbox[1, 1],
            extendbox[2, 0] : extendbox[2, 1],
        ]
        sliceim = sliceim2[np.newaxis, ...]

        output_name = os.path.join(prep_folder, name + "_clean")
        output_name = output_name.replace(".nii.gz", "")
        np.save(output_name, sliceim)

    except:
        logging.error("bug in " + name)
        raise
    logging.info(name + " done")


def full_prep(data_path, prep_folder, n_worker=None, use_existing=True):
    """
    Preprocesses the data by saving numpy arrays.

    Args:
        data_path (str): Path to the data folder.
        prep_folder (str): Path to the folder where the preprocessed data will be saved.
        n_worker (int, optional): Number of worker processes to use for parallel processing. Defaults to None.
        use_existing (bool, optional): Flag indicating whether to use existing preprocessed data if available. Defaults to True.

    Returns:
        list: List of file names in the data folder.

    """
    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)

    logging.info("starting preprocessing")
    pool = Pool(n_worker)
    filelist = [f for f in os.listdir(data_path)]
    partial_savenpy = partial(
        savenpy,
        filelist=filelist,
        prep_folder=prep_folder,
        data_path=data_path,
        use_existing=use_existing,
    )

    N = len(filelist)
    _ = pool.map(partial_savenpy, range(N))
    pool.close()
    pool.join()
    logging.info("end preprocessing")
    return filelist


if __name__ == "__main__":
    orig_dir = (
        "/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/data/remaining_best_niftis"
    )
    best_niftis = pd.read_csv(
        "/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/data/remaining_best_niftis.csv"
    )
    pid_list = best_niftis["pid"].unique()

    for pid in pid_list:
        try:
            pid = str(pid)
            src_folder = os.path.join(orig_dir, pid)
            prep_folder = (
                "/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/data/remaining_segmented_lungs/"
                + pid
            )

            if not os.path.exists(prep_folder):
                os.mkdir(prep_folder)

            logging.info(f"src_folder: {src_folder}")
            logging.info(f"destination folder: {prep_folder}")
            full_prep(src_folder, prep_folder, n_worker=None, use_existing=False)

        except Exception as e:
            logging.error(repr(e))
