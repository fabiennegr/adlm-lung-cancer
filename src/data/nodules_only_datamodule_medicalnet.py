import os
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from monai.transforms import Compose, RandAffine, ScaleIntensityRange, ToTensor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


def reshape_nodule(nodule, new_shape=(41, 41, 16)):
    """
    Reshapes a nodule to a specified shape.

    If the nodule has a shape smaller than the specified shape, it will be padded with -1024.
    If the nodule has a shape larger than the specified shape, it will be center-cropped.

    Args:
        nodule (ndarray): The nodule to be reshaped.
        new_shape (tuple): The desired shape of the nodule. Defaults to (41, 41, 16).

    Returns:
        ndarray: The reshaped nodule.
    """
    shape = nodule.shape

    for i in range(3):
        if shape[i] < new_shape[i]:
            pad = (new_shape[i] - shape[i]) // 2
            remaining = new_shape[i] - shape[i] - pad

            pad_config = [(0, 0)] * 3
            pad_config[i] = (pad, remaining)

            nodule = np.pad(nodule, pad_config, mode="constant", constant_values=-1024)

        elif shape[i] > new_shape[i]:
            crop = (shape[i] - new_shape[i]) // 2
            nodule = np.take(nodule, range(crop, crop + new_shape[i]), axis=i)

    return nodule


class NoduleDataset(Dataset):
    """
    Dataset class for loading and processing nodule data.

    Args:
        df (pandas.DataFrame): The dataframe containing the nodule data.
        transform (callable, optional): A function/transform to be applied on the nodule data. Default is None.

    Returns:
        tuple: A tuple containing the first three nodules, label.

    """

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        nodule_path = self.df.iloc[idx]["nodule_path"]
        label = torch.tensor(self.df.iloc[idx]["lung_cancer"]).float().unsqueeze(0)

        # Load the nodules
        nodules_files = sorted(
            os.listdir(nodule_path), key=lambda x: int(x.split("$")[-1].split(".")[0])
        )
        nodules_files = nodules_files[:3]  # Keep only the first three nodules

        nodules = []
        for file in nodules_files:
            if file.endswith(".npy"):
                nodule = np.load(os.path.join(nodule_path, file))
                # print(f"nodule shape0: {nodule.shape}")

                nodule = reshape_nodule(nodule, new_shape=(41, 41, 16))
                # print(f"nodule shape1: {nodule.shape}")

                if self.transform:
                    nodule = self.transform(nodule)

                # print(f"nodule shape2: {nodule.shape}")

                # add 1 channel dimension
                nodule = nodule.unsqueeze(0)

                # print(f"nodule shape3: {nodule.shape}")

                nodule = nodule.to("cpu")
                nodules.append(nodule)

        # Stack the nodules for batch processing
        return (nodules[0], nodules[1], nodules[2], label)


class NoduleDataModule(LightningDataModule):
    """
    Data module for loading and preprocessing lung cancer dataset.

    Attributes:
        df (pandas.DataFrame): DataFrame containing dataset information.
        batch_size (int): Batch size for data loading.
        transform (monai.transforms.Compose): Composed transformations for data preprocessing.
    """

    def __init__(
        self,
        csv_path="/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/data/dataset_tcia.csv",
        num_workers=15,
        batch_size=4,
        pin_memory=False,
    ):
        """
        Initializes the NodulesOnlyDataModule.

        Args:
            csv_path (str): Path to the CSV file containing dataset information.
            num_workers (int): Number of workers for data loading.
            batch_size (int): Batch size for data loading.
            pin_memory (bool): Whether to use pinned memory for data loading.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.df = pd.read_csv(csv_path)
        self.df = self.df.loc[self.df["timepoint"] == "T0"]
        self.batch_size = batch_size
        self.transform = Compose(
            [
                ScaleIntensityRange(a_min=-1024, a_max=150, b_min=0.0, b_max=1.0, clip=True),
                # RandAffine(
                #     prob=0.5,
                #     rotate_range=(0.03, 0.03, 0.03),
                #     padding_mode="border",
                # ),
                ToTensor(),
            ]
        )

    def setup(self, stage=None):
        """
        Set up the data module by splitting the data into training, validation, and testing sets.

        Args:
            stage: Optional. Specifies the stage of setup (e.g., 'fit', 'test', 'predict').

        Returns:
            None
        """
        # Split the data
        train_test_split_df = pd.read_csv(
            "/local_ssd/practical_wise24/lung_cancer/utils_by_johannes/matched_ids/age_gender_matched_tuples.csv"
        )

        # Random shuffle the rows of the dataframe
        train_test_split_df = train_test_split_df.sample(frac=1, random_state=42).reset_index(
            drop=True
        )

        # Use 70% of the data for training, 20% for validation and 10% for testing
        train_split_df = train_test_split_df[: int(0.7 * len(train_test_split_df))]
        val_split_df = train_test_split_df[
            int(0.7 * len(train_test_split_df)) : int(0.9 * len(train_test_split_df))
        ]
        test_split_df = train_test_split_df[int(0.9 * len(train_test_split_df)) :]

        list_train_patients = (
            train_split_df["cancer"].tolist()
            + train_split_df["no_cancer_negative_scan"].tolist()
            + train_split_df["no_cancer_positive_scan"].tolist()
        )
        list_val_patients = (
            val_split_df["cancer"].tolist()
            + val_split_df["no_cancer_negative_scan"].tolist()
            + val_split_df["no_cancer_positive_scan"].tolist()
        )
        list_test_patients = (
            test_split_df["cancer"].tolist()
            + test_split_df["no_cancer_negative_scan"].tolist()
            + test_split_df["no_cancer_positive_scan"].tolist()
        )

        train_df = self.df.loc[self.df["patient_id"].isin(list_train_patients)]
        val_df = self.df.loc[self.df["patient_id"].isin(list_val_patients)]
        test_df = self.df.loc[self.df["patient_id"].isin(list_test_patients)]

        self.train_dataset = NoduleDataset(train_df, transform=self.transform)
        self.val_dataset = NoduleDataset(val_df, transform=self.transform)
        self.test_dataset = NoduleDataset(test_df, transform=self.transform)

    def train_dataloader(self):
        """
        Returns a DataLoader object for training the model.

        Returns:
            DataLoader: The DataLoader object that loads the training dataset in batches.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Returns a DataLoader object for the validation dataset.

        Returns:
            DataLoader: A DataLoader object that loads the validation dataset in batches.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Returns a DataLoader object for the test dataset.

        Returns:
            DataLoader: The DataLoader object for the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = NoduleDataModule()
