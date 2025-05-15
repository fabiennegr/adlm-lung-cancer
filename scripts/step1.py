# %%
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dicom
import scipy.ndimage
from skimage import measure, morphology

# %%


def load_scan(path):
    """
    Load DICOM scans from the specified path.

    Args:
        path (str): The path to the directory containing the DICOM files.

    Returns:
        list: A list of DICOM slices, sorted by their position in the patient's body.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path '{path}' does not exist.")

    slices = [dicom.read_file(path + "/" + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num + 1
        slice_num = int(len(slices) / sec_num)
        slices.sort(key=lambda x: float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
        )
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def load_scan_nifti(path):
    """
    Load a NIfTI file from the given path and return the image data and NIfTI object.

    Args:
        path (str): The path to the NIfTI file.

    Returns:
        tuple: A tuple containing the image data as a numpy array and the NIfTI object.

    """
    nifti = nib.load(path)
    image = nifti.get_fdata()

    return image, nifti


def get_pixels_hu(image, nifti):
    """
    Convert the Hounsfield units (HU) of a 3D image to int16 and return the converted image along with the pixel dimensions.

    Parameters:
    image (numpy.ndarray): The 3D image array.
    nifti (nibabel.nifti1.Nifti1Image): The NIfTI image object.

    Returns:
    numpy.ndarray: The converted image array with HU values as int16.
    numpy.ndarray: The pixel dimensions as a 1D array, including the slice thickness and pixel spacing.
    """
    # image = np.stack([s.pixel_array for s in slices])
    image = np.transpose(image, (2, 0, 1))
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    return np.array(image, dtype=np.int16), np.array(
        [nifti.header["pixdim"][3]] + [nifti.header["pixdim"][1], nifti.header["pixdim"][2]],
        dtype=np.float32,
    )


def binarize_per_slice(
    image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10
):
    """
    Binarizes each slice of a 3D image based on intensity and region properties.

    Args:
        image (ndarray): The 3D image to be binarized.
        spacing (tuple): The spacing between pixels in each dimension.
        intensity_th (float, optional): The intensity threshold for binarization. Defaults to -600.
        sigma (float, optional): The standard deviation of the Gaussian filter. Defaults to 1.
        area_th (float, optional): The minimum region area threshold. Defaults to 30.
        eccen_th (float, optional): The maximum eccentricity threshold. Defaults to 0.99.
        bg_patch_size (int, optional): The size of the background patch. Defaults to 10.

    Returns:
        ndarray: The binarized image with the same shape as the input image.
    """
    bw = np.zeros(image.shape, dtype=bool)

    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size / 2 + 0.5, image_size / 2 - 0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2 + y**2) ** 0.5
    nan_mask = (d < image_size / 2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = (
                scipy.ndimage.filters.gaussian_filter(
                    np.multiply(image[i].astype("float32"), nan_mask), sigma, truncate=2.0
                )
                < intensity_th
            )
        else:
            current_bw = (
                scipy.ndimage.filters.gaussian_filter(
                    image[i].astype("float32"), sigma, truncate=2.0
                )
                < intensity_th
            )

        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw

    return bw


def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    """
    Perform analysis on all slices of a binary image.

    Args:
        bw (ndarray): Binary image.
        spacing (ndarray): Spacing between pixels in each dimension.
        cut_num (int, optional): Number of top layers to be removed. Defaults to 0.
        vol_limit (list, optional): Volume limits for component selection. Defaults to [0.68, 8.2].
        area_th (float, optional): Area threshold for component selection. Defaults to 6e3.
        dist_th (float, optional): Distance threshold for component selection. Defaults to 62.

    Returns:
        tuple: A tuple containing the following:
            - bw (ndarray): Updated binary image after analysis.
            - num_valid_labels (int): Number of valid labels after analysis.
    """
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set(
        [
            label[0, 0, 0],
            label[0, 0, -1],
            label[0, -1, 0],
            label[0, -1, -1],
            label[-1 - cut_num, 0, 0],
            label[-1 - cut_num, 0, -1],
            label[-1 - cut_num, -1, 0],
            label[-1 - cut_num, -1, -1],
            label[0, 0, mid],
            label[0, -1, mid],
            label[-1 - cut_num, 0, mid],
            label[-1 - cut_num, -1, mid],
        ]
    )
    for l in bg_label:
        label[label == l] = 0

    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if (
            prop.area * spacing.prod() < vol_limit[0] * 1e6
            or prop.area * spacing.prod() > vol_limit[1] * 1e6
        ):
            label[label == prop.label] = 0

    # prepare a distance map for further analysis
    x_axis = (
        np.linspace(-label.shape[1] / 2 + 0.5, label.shape[1] / 2 - 0.5, label.shape[1])
        * spacing[1]
    )
    y_axis = (
        np.linspace(-label.shape[2] / 2 + 0.5, label.shape[2] / 2 - 0.5, label.shape[2])
        * spacing[2]
    )
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2 + y**2) ** 0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))

        if (
            np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th])
            < dist_th
        ):
            valid_label.add(vol.label)

    bw = np.in1d(label, list(valid_label)).reshape(label.shape)

    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label == l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)

    return bw, len(valid_label)


def fill_hole(bw):
    """
    Fills 3D holes in a binary image.

    Parameters:
    - bw: numpy.ndarray
        Binary image where holes need to be filled.

    Returns:
    - bw: numpy.ndarray
        Binary image with filled holes.
    """
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set(
        [
            label[0, 0, 0],
            label[0, 0, -1],
            label[0, -1, 0],
            label[0, -1, -1],
            label[-1, 0, 0],
            label[-1, 0, -1],
            label[-1, -1, 0],
            label[-1, -1, -1],
        ]
    )
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)

    return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):
    """
    Extracts the main regions of the lungs from a binary image.

    Args:
        bw (ndarray): Binary image of the lungs.
        spacing (float): Spacing between pixels in the image.
        max_iter (int, optional): Maximum number of iterations for finding the main regions. Defaults to 22.
        max_ratio (float, optional): Maximum ratio between the areas of the two largest regions. Defaults to 4.8.

    Returns:
        tuple: A tuple containing three binary images:
            - bw1: Binary image of the first lung region.
            - bw2: Binary image of the second lung region.
            - bw: Binary image with both lung regions.
    """

    def extract_main(bw, cover=0.95):
        """
        Extracts the main regions from a binary image.

        Args:
            bw (ndarray): Binary image.
            cover (float, optional): The desired coverage of the main regions. Defaults to 0.95.

        Returns:
            ndarray: Binary image with only the main regions.
        """
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area) * cover:
                sum = sum + area[count]
                count = count + 1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0] : bb[2], bb[1] : bb[3]] = (
                    filter[bb[0] : bb[2], bb[1] : bb[3]] | properties[j].convex_image
                )
            bw[i] = bw[i] & filter

        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label == properties[0].label

        return bw

    def fill_2d_hole(bw):
        """
        Fills 2D holes in a binary image slice by slice.

        Args:
            bw (ndarray): Binary image represented as a 2D NumPy array.

        Returns:
            ndarray: Binary image with 2D holes filled.

        """
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0] : bb[2], bb[1] : bb[3]] = (
                    current_slice[bb[0] : bb[2], bb[1] : bb[3]] | prop.filled_image
                )
            bw[i] = current_slice

        return bw

    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area / properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1

    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)

        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)

    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype("bool")

    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw


def step1_python(case_path):
    """
    Preprocesses the lung CT scan data.

    Args:
        case_path (str): The path to the CT scan data.

    Returns:
        tuple: A tuple containing the preprocessed data, including:
            - case_pixels (numpy.ndarray): The pixel values of the CT scan.
            - bw1 (numpy.ndarray): The binary mask of the left lung.
            - bw2 (numpy.ndarray): The binary mask of the right lung.
            - spacing (list): The pixel spacing of the CT scan.
            - nifti (nibabel.nifti1.Nifti1Image): The NIfTI image object.

    """
    # case = load_scan(case_path)
    image, nifti = load_scan_nifti(case_path)
    case_pixels, spacing = get_pixels_hu(image, nifti)
    bw = binarize_per_slice(case_pixels, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68, 7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return case_pixels, bw1, bw2, spacing, nifti


if __name__ == "__main__":
    # INPUT_FOLDER = '/work/DataBowl3/stage1/stage1/'
    INPUT_FOLDER = "/home/brandtj/Documents/projects/iderha/nlst/converted_niftis/100570"
    patients = list(str(INPUT_FOLDER).split("/"))[-2]
    # patients.sort()
    # case_pixels, m1, m2, spacing = step1_python(os.path.join(INPUT_FOLDER, patients[0]))
    case_pixels, m1, m2, spacing = step1_python(INPUT_FOLDER)

    # %%

    plt.imshow(m1[60])
    plt.show()
    plt.figure()
    plt.imshow(m2[60])
    plt.show()
#     first_patient = load_scan(INPUT_FOLDER + patients[25])
#     first_patient_pixels, spacing = get_pixels_hu(first_patient)
#     plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
#     plt.xlabel("Hounsfield Units (HU)")
#     plt.ylabel("Frequency")
#     plt.show()

#     # Show some slice in the middle
#     h = 80
#     plt.imshow(first_patient_pixels[h], cmap=plt.cm.gray)
#     plt.show()

#     bw = binarize_per_slice(first_patient_pixels, spacing)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()

#     flag = 0
#     cut_num = 0
#     while flag == 0:
#         bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num)
#         cut_num = cut_num + 1
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()

#     bw = fill_hole(bw)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()

#     bw1, bw2, bw = two_lung_only(bw, spacing)
#     plt.imshow(bw[h], cmap=plt.cm.gray)
#     plt.show()
