import os
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from monai.transforms import Compose, ResizeWithPadOrCrop, ScaleIntensityRange, ToTensor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


def reshape_nodule(nodule, new_shape=(41, 41, 16)):
    """
    Reshapes a nodule to a specified shape.

    If the nodule has a shape smaller than the specified shape, it will be padded with -1024 to match the new shape.
    If the nodule has a shape larger than the specified shape, it will be center-cropped to match the new shape.

    Args:
        nodule (ndarray): The nodule to be reshaped.
        new_shape (tuple, optional): The desired shape of the nodule. Defaults to (41, 41, 16).

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


class MultiModalDataset(Dataset):
    """
    Dataset class for handling multimodal data.

    Args:
        df (pandas.DataFrame): The input dataframe containing the dataset.
        transform (callable, optional): Optional transform to be applied on the nodules.
        image_transform (callable, optional): Optional transform to be applied on the original image.

    Attributes:
        df (pandas.DataFrame): The input dataframe containing the dataset.
        transform (callable, optional): Optional transform to be applied on the nodules.
        image_transform (callable, optional): Optional transform to be applied on the original image.
        cat_features (list): List of categorical features in the dataset.
        preprocessor_tabular (sklearn.compose.ColumnTransformer): Preprocessor for tabular features.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.

    """

    def __init__(self, df, transform=None, image_transform=None):
        """
        Initialize the MultimodalDataset class.

        Args:
            df (pandas.DataFrame): The input dataframe containing the dataset.
            transform (callable, optional): A function/transform to be applied on the non-image data. Default is None.
            image_transform (callable, optional): A function/transform to be applied on the image data. Default is None.
        """
        self.df = df
        self.transform = transform
        self.image_transform = image_transform

        df_tabular = self.df[
            [
                "age",
                "educat",
                "ethnic",
                "gender",
                "race",
                "diagcopd",
                "height",
                "weight",
                "smokeage",
                "pkyr",
                "smokeday",
                "smokeyr",
                "cigsmok",
            ]
        ]

        num_features = [
            "age",
            "educat",
            "height",
            "weight",
            "smokeage",
            "pkyr",
            "smokeday",
            "smokeyr",
            "cigsmok",
        ]
        self.cat_features = ["ethnic", "gender", "race", "diagcopd"]

        num_transform = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        cat_transform = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.preprocessor_tabular = ColumnTransformer(
            transformers=[
                ("num", num_transform, num_features),
                ("cat", cat_transform, self.cat_features),
            ]
        )

        self.preprocessor_tabular.fit(df_tabular)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the tabular data, original image, nodules, and label.

        """
        nodule_path = self.df.iloc[idx]["nodule_path"]
        original_image_path = self.df.iloc[idx]["original_image"]

        tabular_features = self.df.iloc[idx][
            [
                "age",
                "educat",
                "ethnic",
                "gender",
                "race",
                "diagcopd",
                "height",
                "weight",
                "smokeage",
                "pkyr",
                "smokeday",
                "smokeyr",
                "cigsmok",
            ]
        ]

        tabular_features[self.cat_features] = tabular_features[self.cat_features].astype(str)
        tabular_features = self.preprocessor_tabular.transform(tabular_features.to_frame().T)
        tabular_data = torch.from_numpy(tabular_features).float()

        original_image = nib.load(original_image_path)
        original_image = original_image.get_fdata()

        original_image = original_image[None]

        if self.image_transform:
            original_image = self.image_transform(original_image)

        original_image = original_image.repeat(3, 1, 1, 1)

        original_image = original_image.to("cpu")

        label = torch.tensor(self.df.iloc[idx]["lung_cancer"]).float().unsqueeze(0)

        nodules_files = sorted(
            os.listdir(nodule_path), key=lambda x: int(x.split("$")[-1].split(".")[0])
        )
        nodules_files = nodules_files[:3]

        nodules = []
        for file in nodules_files:
            if file.endswith(".npy"):
                nodule = np.load(os.path.join(nodule_path, file))
                nodule = reshape_nodule(nodule, new_shape=(41, 41, 16))

                if self.transform:
                    nodule = self.transform(nodule)

                nodule = nodule.unsqueeze(0)
                nodule = nodule.to("cpu")
                nodules.append(nodule)

        return (tabular_data, original_image, nodules[0], nodules[1], nodules[2], label)


class MultiModalDataModule(LightningDataModule):
    """
    Data module for handling multi-modal data in a PyTorch Lightning project.

    Args:
        csv_path (str): The path to the CSV file containing the dataset.
        num_workers (int): The number of workers for data loading.
        batch_size (int): The batch size for training, validation, and testing.
        pin_memory (bool): Whether to pin memory for faster data transfer.

    Attributes:
        df (pandas.DataFrame): The DataFrame containing the dataset.
        batch_size (int): The batch size for training, validation, and testing.
        transform (torchvision.transforms.Compose): The transform applied to the data.
        image_transform (torchvision.transforms.Compose): The transform applied to the images.
        train_dataset (MultiModalDataset): The training dataset.
        val_dataset (MultiModalDataset): The validation dataset.
        test_dataset (MultiModalDataset): The testing dataset.
    """

    def __init__(
        self,
        csv_path="/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/data/dataset_all.csv",
        num_workers=15,
        batch_size=4,
        pin_memory=False,
    ):
        """
        Initializes the MultimodalDataset class.

        Args:
            csv_path (str, optional): Path to the CSV file containing the dataset. Defaults to "/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/data/dataset_all.csv".
            num_workers (int, optional): Number of worker processes for data loading. Defaults to 15.
            batch_size (int, optional): Number of samples per batch. Defaults to 4.
            pin_memory (bool, optional): If True, the data loader will copy Tensors into pinned memory. Defaults to False.
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

        self.image_transform = Compose(
            [
                ResizeWithPadOrCrop(
                    spatial_size=(400, 400, 120), method="symmetric", mode="minimum"
                ),
                ScaleIntensityRange(a_min=-1024, a_max=150, b_min=0.0, b_max=1.0, clip=True),
                ToTensor(),
            ]
        )

    def setup(self, stage=None):
        """
        Set up the data for training, validation, and testing.

        Args:
            stage (str, optional): Stage of the setup. Defaults to None.

        Returns:
            None
        """
        # Split the data
        train_test_split_df = pd.read_csv(
            "/local_ssd/practical_wise24/lung_cancer/utils_by_johannes/matched_ids/age_gender_matched_tuples.csv"
        )

        # random shuffle the rows of the dataframe
        train_test_split_df = train_test_split_df.sample(frac=1, random_state=42).reset_index(
            drop=True
        )

        # use 70% of the data for training, 20% for validation and 10% for testing
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

        self.train_dataset = MultiModalDataset(
            train_df, transform=self.transform, image_transform=self.image_transform
        )
        self.val_dataset = MultiModalDataset(
            val_df, transform=self.transform, image_transform=self.image_transform
        )
        self.test_dataset = MultiModalDataset(
            test_df, transform=self.transform, image_transform=self.image_transform
        )

    def train_dataloader(self):
        """
        Returns a DataLoader object for training the model.

        Returns:
            DataLoader: A DataLoader object that loads the training dataset in batches.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Returns a DataLoader object for the validation dataset.

        Returns:
            DataLoader: A DataLoader object that provides batches of data from the validation dataset.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Returns a DataLoader object for the test dataset.

        Returns:
            DataLoader: A DataLoader object that batches the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        Args:
            stage (str, optional): The stage being torn down. Either "fit", "validate", "test", or "predict".
                Defaults to None.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """
        Called when saving a checkpoint. Implement to generate and save the datamodule state.

        Returns:
            Dict[Any, Any]: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        Args:
            state_dict (Dict[str, Any]): The datamodule state returned by `self.state_dict()`.

        Returns:
            None
        """
        pass
