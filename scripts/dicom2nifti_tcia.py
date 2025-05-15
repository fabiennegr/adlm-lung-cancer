import logging
import os
import shutil as sh
import zipfile
from pathlib import Path

import dicom2nifti
import dicom2nifti.settings as settings
import pandas as pd
import tqdm

logging.basicConfig(
    filename="/local_ssd/practical_wise24/lung_cancer/utils_by_johannes/logs/out.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

logging.basicConfig(
    filename="/local_ssd/practical_wise24/lung_cancer/utils_by_johannes/logs/err.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def unzip_file(zip_path, destination):
    """
    Unzips a file to the specified destination.

    Args:
        zip_path (str): The path to the zip file.
        destination (str): The destination directory to extract the files to.

    Returns:
        None
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination)
        logging.info(f"Extracted {zip_path} to {destination}")


def get_zip_files(path):
    """
    Get a list of zip files in the specified path.

    Args:
        path (str): The path to search for zip files.

    Returns:
        list: A list of zip file names.

    """
    path_obj = Path(path)
    zip_files = [file.name for file in path_obj.glob("*.zip") if file.is_file()]
    return zip_files


def unpack_and_remove_first_zip(zip_files_list, target_dir):
    """
    Unpacks the first zip file from the given list and removes it from the target directory.

    Args:
        zip_files_list (list): A list of zip file names.
        target_dir (str): The target directory where the zip file is located.

    Returns:
        None
    """
    if not zip_files_list:
        logging.info("No zip files in the list.")
        return

    first_zip_file = zip_files_list[0]
    zip_path = os.path.join(target_dir, first_zip_file)

    if not os.path.exists(zip_path):
        logging.info(f"Zip file {first_zip_file} not found in the target directory.")
        return

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    os.remove(zip_path)
    zip_files_list.pop(0)

    return logging.info(f"Unpacked and removed {first_zip_file}")


def is_folder_empty(folder_path):
    """
    Check if the given folder is empty.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        bool: True if the folder is empty, False otherwise.
    """
    return os.path.isdir(folder_path) and not os.listdir(folder_path)


def find_parent_folders_with_dcm(folder_path):
    """
    Find parent folders that contain DICOM files within the given folder path.

    Args:
        folder_path (str): The path to the folder to search for DICOM files.

    Returns:
        list: A list of parent folder paths that contain DICOM files.
    """
    folder = Path(folder_path)
    parent_folders = set()

    for dcm_file in folder.glob("**/*.dcm"):
        parent_folders.add(dcm_file.parent)

    return list(parent_folders)


def find_nfiti_file(folder_path):
    """
    Find all .nii.gz files recursively in the given folder path.

    Args:
        folder_path (str): The path to the folder to search for .nii.gz files.

    Returns:
        tuple: A tuple containing a list of file paths and the input folder path.
    """
    folder = Path(folder_path)
    file_name = list(folder.glob("**/*.nii.gz"))
    return file_name, folder_path


def rename_file(old_name, new_name, definitive_save_dir):
    """
    Renames a file from the old name to the new name.

    Args:
        old_name (str): The current name of the file.
        new_name (str): The desired new name for the file.
        definitive_save_dir (str): The directory where the file will be saved.

    Raises:
        OSError: If renaming the file is not possible.
        ValueError: If the old_name list is empty.
        ValueError: If the old_name list contains more than one element.

    Returns:
        None
    """
    if len(old_name) == 1:
        try:
            os.rename(old_name[0], new_name)
        except Exception as e:
            logging.error(f"Renaming not possible {new_name} \n {e}")
    elif len(old_name) == 0:
        logging.error(f"Nifti does not seem to exist {new_name}")
    else:
        logging.info("Script does not work as expected.")
        logging.error(f"More than one Nifti {new_name}")


def convert_dicom_files(dicom, output_dir, error_dicoms):
    """
    Convert DICOM files to NIfTI format.

    Args:
        dicom (str): Path to the DICOM directory.
        output_dir (str): Path to the output directory where the NIfTI files will be saved.
        error_dicoms (list): List to store DICOM files that encountered errors during conversion.

    Raises:
        Exception: If an error occurs during the conversion process.

    Returns:
        None
    """
    try:
        dicom2nifti.convert_directory(dicom, output_dir, compression=True, reorient=True)
    except Exception:
        try:
            settings.disable_validate_slice_increment()
            settings.enable_resampling()
            settings.set_resample_spline_interpolation_order(1)
            settings.set_resample_padding(-1000)
            dicom2nifti.convert_directory(dicom, output_dir, compression=True, reorient=True)
            settings.enable_validate_slice_increment()
            settings.disable_resampling()
        except Exception as e:
            logging.info(Exception(f"Error with {dicom} \n {e}"))
            logging.error(f"{dicom}: {e}")


def clear_directory(folder_path):
    """
    Clears the contents of a directory by removing all files and subdirectories.

    Args:
        folder_path (str): The path to the directory to be cleared.

    Returns:
        None
    """
    if not os.path.exists(folder_path):
        logging.info(f"The folder {folder_path} does not exist.")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove files and links
            elif os.path.isdir(item_path):
                sh.rmtree(item_path)  # Remove directories
            logging.info(f"Removed {item_path}")
        except Exception as e:
            logging.info(f"Failed to remove {item_path}. Reason: {e}")
            logging.error(f"{item_path}: {e}")


if __name__ == "__main__":
    df = pd.read_csv(
        "/local_ssd/practical_wise24/lung_cancer/tabular_data/Spiral CT Image Information/sct_image_series_d040722.csv"
    )

    df_pids = pd.read_csv(
        "/local_ssd/practical_wise24/lung_cancer/utils_by_johannes/matched_ids/age_gender_matched_tuples.csv"
    )
    all_pids = df.stack().tolist()

    error_dicoms = []
    skipped_dicoms = []

    root_TCIA_dir = "/local_ssd/practical_wise24/lung_cancer/images/images_TCIA/"

    decoy_save_dir = "/local_ssd/practical_wise24/lung_cancer/images/nifti/decoy"

    if not os.path.exists(decoy_save_dir):
        os.makedirs(decoy_save_dir, exist_ok=True)

    participant_level = [
        os.path.join(root_TCIA_dir, name)
        for name in os.listdir(root_TCIA_dir)
        if os.path.isdir(os.path.join(root_TCIA_dir, name))
    ]

    for participant in participant_level:
        timepoint_level = [
            os.path.join(participant, name)
            for name in os.listdir(participant)
            if os.path.isdir(os.path.join(participant, name))
        ]

        for time_idx, timepoint_folder in enumerate(timepoint_level):
            replica_level = [
                os.path.join(timepoint_folder, name)
                for name in os.listdir(timepoint_folder)
                if os.path.isdir(os.path.join(timepoint_folder, name))
            ]

            for replica_idx, replica in enumerate(replica_level):
                dicom_dir = find_parent_folders_with_dcm(replica)
                folder_to_clear = decoy_save_dir
                clear_directory(folder_to_clear)

                for dicom in tqdm.tqdm(dicom_dir):
                    assert is_folder_empty(
                        decoy_save_dir
                    ), f"The folder {decoy_save_dir} is not empty."
                    logging.warning(f"The folder is not empty at dicom: {dicom}")
                    split_path = str(dicom).split("/")
                    pid = split_path[-3]

                    if int(pid) in all_pids:
                        seriesinstanceuid = split_path[-1]
                        timepoint = df[
                            (df["pid"] == int(pid))
                            & (df["seriesinstanceuid"] == seriesinstanceuid)
                        ]["study_yr"].values[0]

                        save_dir = f"/local_ssd/practical_wise24/lung_cancer/images/converted_nifti_TCIA/{pid}"

                        if (
                            len(
                                df.loc[
                                    ~df["imagetype"].str.contains("LOCALIZER")
                                    & (df["pid"] == int(pid))
                                    & (df["seriesinstanceuid"] == seriesinstanceuid)
                                    & (df["reconfilter"] == "LUNG")
                                    & (df["numberimages"] > 3)
                                ]
                            )
                            == 1
                        ):
                            slice_thickness = df.loc[
                                ~df["imagetype"].str.contains("LOCALIZER")
                                & (df["pid"] == int(pid))
                                & (df["seriesinstanceuid"] == seriesinstanceuid)
                                & (df["numberimages"] > 3)
                            ]["reconthickness"].values[0]
                            studyuid = df.loc[
                                ~df["imagetype"].str.contains("LOCALIZER")
                                & (df["pid"] == int(pid))
                                & (df["seriesinstanceuid"] == seriesinstanceuid)
                                & (df["numberimages"] > 3)
                            ]["studyuid"].values[0]

                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir, exist_ok=True)

                            convert_dicom_files(dicom, decoy_save_dir, error_dicoms)
                            old_name, old_dir = find_nfiti_file(decoy_save_dir)
                            new_name = f"{save_dir}/{pid}${studyuid}${seriesinstanceuid}$LUNG${slice_thickness}${timepoint}_0000.nii.gz"
                            rename_file(old_name, new_name, save_dir)

                        elif (
                            len(
                                df.loc[
                                    ~df["imagetype"].str.contains("LOCALIZER")
                                    & (df["pid"] == int(pid))
                                    & (df["seriesinstanceuid"] == seriesinstanceuid)
                                    & (df["numberimages"] > 3)
                                ]
                            )
                            == 1
                        ):
                            reconfilter = df.loc[
                                ~df["imagetype"].str.contains("LOCALIZER")
                                & (df["pid"] == int(pid))
                                & (df["seriesinstanceuid"] == seriesinstanceuid)
                                & (df["numberimages"] > 3)
                            ]["reconfilter"].values[0]
                            slice_thickness = df.loc[
                                ~df["imagetype"].str.contains("LOCALIZER")
                                & (df["pid"] == int(pid))
                                & (df["seriesinstanceuid"] == seriesinstanceuid)
                                & (df["numberimages"] > 3)
                            ]["reconthickness"].values[0]
                            studyuid = df.loc[
                                ~df["imagetype"].str.contains("LOCALIZER")
                                & (df["pid"] == int(pid))
                                & (df["seriesinstanceuid"] == seriesinstanceuid)
                                & (df["numberimages"] > 3)
                            ]["studyuid"].values[0]

                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir, exist_ok=True)

                            convert_dicom_files(dicom, decoy_save_dir, error_dicoms)
                            old_name, old_dir = find_nfiti_file(decoy_save_dir)
                            new_name = f"{save_dir}/{pid}${studyuid}${seriesinstanceuid}${reconfilter}${slice_thickness}${timepoint}_0000.nii.gz"
                            rename_file(old_name, new_name, save_dir)

                        elif (
                            len(
                                df.loc[
                                    ~df["imagetype"].str.contains("LOCALIZER")
                                    & (df["pid"] == int(pid))
                                    & (df["seriesinstanceuid"] == seriesinstanceuid)
                                    & (df["numberimages"] > 3)
                                ]
                            )
                            == 0
                        ):
                            continue

                        else:
                            skipped_dicoms.append(dicom)
                            logging.warning(f"{dicom}: There was more than one file.")

                        clear_directory(folder_to_clear)

    logging.info("Skipped dicoms: ", skipped_dicoms)
    logging.info("Error dicoms: ", error_dicoms)
