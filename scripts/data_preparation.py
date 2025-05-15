import logging
import os
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import tqdm
from full_prep_lung_mask import full_prep

logging.basicConfig(
    filename="/local_ssd/practical_wise24/lung_cancer/prasanga/logs/data_prep.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

logging.basicConfig(
    filename="/local_ssd/practical_wise24/lung_cancer/prasanga/logs/data_prep_err.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


class NiftiPreprocessor:
    """
    Preprocesses Nifti files for lung cancer prediction.

    Args:
        root_data_folder (str): Root folder containing the input data.
        prediction_root_folder (str): Root folder containing the prediction files.
        mask_root_folder (str): Root folder containing the lung mask files.
        nodule_root_folder (str): Root folder to save the cropped nodule images.
        file_path (str): Path to the Nifti file.

    Attributes:
        root_data_folder (Path): Root folder containing the input data.
        prediction_root_folder (Path): Root folder containing the prediction files.
        mask_root_folder (Path): Root folder containing the lung mask files.
        nodule_root_folder (Path): Root folder to save the cropped nodule images.
        file_path (str): Path to the Nifti file.
        file_name (str): Name of the Nifti file.

    Methods:
        get_matching_prediction: Get the path to the matching prediction file.
        get_matching_lung_mask: Get the path to the matching lung mask file.
        filter_prediction: Filter the prediction based on the lung mask.
        get_cropped_images: Get the cropped nodule images.
        save_cropped_images: Save the cropped nodule images to the specified folder.
    """

    def get_matching_prediction(self):
        """
        Get the path of the matching prediction file based on the input file name.

        Returns:
            The path of the matching prediction file if it exists, None otherwise.
        """
        expected_prediction_name = self.file_name.replace(
            "_0000.nii.gz", "_boxes.pkl"
        )  # Converts "image.nii.gz" to "image_boxes.pkl"
        expected_prediction_path = self.prediction_root_folder / expected_prediction_name

        if expected_prediction_path.exists():
            return expected_prediction_path
        else:
            logging.error(f"No prediction found for case: {self.file_name}")
            return None

    def get_matching_lung_mask(self):
        """
        Retrieves the path of the matching lung mask file for the current file.

        Returns:
            The path of the matching lung mask file if it exists, otherwise None.
        """
        expected_mask_name = self.file_name.replace(".nii.gz", "_mask.npy")
        expected_mask_path = self.mask_root_folder / expected_mask_name

        if expected_mask_path.exists():
            return expected_mask_path
        else:
            logging.error(f"No lung mask found for prediction: {self.file_name}")
            return None

    def filter_prediction(self):
        """
        Filter the prediction dictionary based on the intersection of bounding boxes with a lung mask.

        Returns:
            dict: A filtered prediction dictionary containing the top predictions.
        """
        prediction_file = self.get_matching_prediction()
        lung_mask_path = self.get_matching_lung_mask()
        if prediction_file and lung_mask_path is not None:
            with open(prediction_file, "rb") as f:
                pred_dict = pickle.load(f)

            lung_mask = np.load(lung_mask_path)

            def intersects_with_mask(box, mask):
                """
                Check if a bounding box intersects with a mask.

                Args:
                    box (list): A list containing the coordinates of the bounding box in the format [z_min, x_min, z_max, x_max, y_min, y_max].
                    mask (ndarray): A 3D numpy array representing the mask.

                Returns:
                    bool: True if any edge point of the bounding box is within the mask, False otherwise.
                """
                # Define the ranges for each axis
                x_range = [int(box[1]), int(box[3])]
                y_range = [int(box[4]), int(box[5])]
                z_range = [int(box[0]), int(box[2])]

                # Create a list of edge points
                edge_points = []
                for x in x_range:
                    for y in y_range:
                        for z in z_range:
                            edge_points.append((x, y, z))

                # Check if any edge points are in the mask
                for x, y, z in edge_points:
                    if (
                        0 <= z < mask.shape[0]
                        and 0 <= y < mask.shape[1]
                        and 0 <= x < mask.shape[2]
                    ):
                        if mask[z, y, x]:
                            return True
                return False

            # 3. Filter boxes
            filtered_boxes_indices = [
                i
                for i, box in enumerate(pred_dict["pred_boxes"])
                if intersects_with_mask(box, lung_mask)
            ]

            # 4. Sort boxes and select top predictions
            sorted_indices = sorted(
                filtered_boxes_indices,
                key=lambda i: pred_dict["pred_scores"][i],
                reverse=True,
            )
            top_indices = sorted_indices[:5]

            # 5. Return filtered prediction dictionary
            filtered_pred_dict = {
                "pred_boxes": [pred_dict["pred_boxes"][i] for i in top_indices],
                "pred_scores": [pred_dict["pred_scores"][i] for i in top_indices],
                "pred_labels": [pred_dict["pred_labels"][i] for i in top_indices],
                "restore": pred_dict["restore"],
                "original_size_of_raw_data": pred_dict["original_size_of_raw_data"],
                "itk_origin": pred_dict["itk_origin"],
                "itk_spacing": pred_dict["itk_spacing"],
                "itk_direction": pred_dict["itk_direction"],
            }

            return filtered_pred_dict

    def get_cropped_images(self):
        """
        Get the cropped nodule images based on the lung mask.

        Returns:
            cropped_images_for_file (list): List of cropped nodule images.
            filtered_preds_dict (dict): Dictionary containing filtered predictions.
        """
        # create the lung mask
        full_prep(str(self.file_path), str(mask_root_folder), n_worker=8, use_existing=False)

        filtered_preds_dict = self.filter_prediction()
        if filtered_preds_dict:
            img = nib.load(self.file_path)
            image_array = img.get_fdata()

            boxes = filtered_preds_dict["pred_boxes"]
            cropped_images_for_file = []

        for box in boxes:
            # PADDING (one slice more for z axis, 3 pixel values in x and y direction)
            cropped_image = image_array[
                int(box[4] - 3) : int(box[5] + 3),  # padding 3 x
                int(box[1] - 3) : int(box[3] + 3),  # padding 3 y
                int(box[0] - 1) : int(box[2] + 1),  # padding 1 z
            ]

            cropped_images_for_file.append(cropped_image)

        return cropped_images_for_file, filtered_preds_dict

    def save_cropped_images(self):
        """
        Saves the cropped images for a given file.

        Returns:
            str: The path to the directory where the cropped images are saved.
                Returns None if no cropped images are found.
        """
        cropped_images_for_file, filtered_preds_dict = self.get_cropped_images()
        if cropped_images_for_file:
            # Extract patient ID and timepoint from filename
            file_name = self.file_name
            patient_id = file_name.split("$")[0]
            timepoint = file_name.split("$")[-1].split("_")[0].replace(".nii.gz", "")

            # Create directories if they don't exist
            patient_dir = os.path.join(self.nodule_root_folder, patient_id)
            if not os.path.exists(patient_dir):
                os.mkdir(patient_dir)

            timepoint_dir = os.path.join(patient_dir, f"T{timepoint}")
            if not os.path.exists(timepoint_dir):
                os.mkdir(timepoint_dir)

            # Sort based on confidence scores
            sorted_indices = sorted(
                range(len(filtered_preds_dict["pred_scores"])),
                key=lambda k: filtered_preds_dict["pred_scores"][k],
                reverse=True,
            )

            for i, idx in enumerate(sorted_indices):
                conf = filtered_preds_dict["pred_scores"][idx]

                new_file_name = file_name.replace("_0000.nii.gz", f"${conf}${idx}")
                img_path = os.path.join(timepoint_dir, new_file_name)

                # iterate over the nodules to extract them
                nodule = cropped_images_for_file[idx]
                np.save(img_path, nodule)

            return timepoint_dir

        else:
            logging.error(f"No cropped images found for file: {self.file_name}")
            return None


def get_all_paths_images(directory):
    """
    Get all paths to NIfTI image files in the specified directory and its subdirectories.

    Args:
        directory (str): The directory to search for NIfTI image files.

    Returns:
        list: A list of pathlib.Path objects representing the paths to the NIfTI image files.
    """
    p = Path(directory).glob("**/*")
    nifti_files = [x for x in p if x.is_file() and (x.name.endswith(".nii.gz"))]
    return nifti_files


if __name__ == "__main__":
    patient_pids = []
    timepoints = []
    original_images = []
    nodules_paths = []

    root_folder = "/local_ssd/practical_wise24/lung_cancer/prasanga/best_niftis/"
    nodule_root_folder = (
        "/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/data/cropped_nodules"
    )
    prediction_root_folder = "/local_ssd/practical_wise24/lung_cancer/prasanga/nnDetection/models_dir/Task016_Luna/Luna16_nodule_detection_epochs_50_default_settings_rodeo_off_deterministic_true/fold2/test_predictions/"
    mask_base_folder = "/local_ssd/practical_wise24/lung_cancer/prasanga/segmented_lungs"

    # iterate over all the folders in the root folder
    count = 0
    pids = os.listdir(root_folder)
    for pid in tqdm.tqdm(pids):
        pid_folder = os.path.join(root_folder, pid)
        nifti_files = get_all_paths_images(pid_folder)
        mask_root_folder = os.path.join(mask_base_folder, pid)

        for file_path in tqdm.tqdm(nifti_files):
            try:
                logging.info(f"Processing file: {file_path}")
                preprocessor = NiftiPreprocessor(
                    pid_folder,
                    prediction_root_folder,
                    mask_root_folder,
                    nodule_root_folder,
                    file_path,
                )
                timepoint_nodule_dir = preprocessor.save_cropped_images()
                timepoint = timepoint_nodule_dir.split("/")[-1]
                patient_pids.append(pid)
                original_images.append(file_path)
                timepoints.append(timepoint)
                nodules_paths.append(timepoint_nodule_dir)

                if count % 10 == 0:
                    df = pd.DataFrame(
                        {
                            "patient_id": patient_pids,
                            "timepoint": timepoints,
                            "original_image": original_images,
                            "nodule_path": nodules_paths,
                        }
                    )

                    df.to_csv(
                        "/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/data.csv",
                        index=False,
                    )

                count += 1

                logging.info(f"Finished processing file: {file_path}")
            except Exception as e:
                logging.error(f"Error processing file: {file_path}")
                logging.error(e)

    df = pd.DataFrame(
        {
            "patient_id": patient_pids,
            "timepoint": timepoints,
            "original_image": original_images,
            "nodule_path": nodules_paths,
        }
    )

    df.to_csv(
        "/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/data/cropped_nodules.csv",
        index=False,
    )

    logging.info("Saving dataframe to csv")
