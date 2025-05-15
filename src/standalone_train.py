import os
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from monai.transforms import Compose, RandAffine, ScaleIntensityRange, ToTensor
from pytorch_lightning import LightningDataModule, LightningModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    BinaryStatScores,
)
from torchvision.transforms import transforms

from models.components.resnet import resnet50


def reshape_nodule(nodule, new_shape=(41, 41, 16)):
    """
    Reshapes a nodule to a specified new shape.

    If the nodule has a shape smaller than the new shape, it will be padded to match the new shape.
    If the nodule has a shape larger than the new shape, it will be center-cropped to match the new shape.

    Args:
        nodule (ndarray): The input nodule to reshape.
        new_shape (tuple, optional): The desired new shape. Defaults to (41, 41, 16).

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
        transform (callable, optional): A function/transform to apply to the nodule data. Default is None.

    Returns:
        tuple: A tuple containing the first three nodules, label.

    """

    def __init__(self, df, transform=None):
        """
        Initialize the class.

        Args:
            df (pandas.DataFrame): The input DataFrame.
            transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
                Default is None.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get the item at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the first three nodules, label.

        """
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
    def __init__(
        self,
        csv_path="/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/data/dataset_tcia.csv",
        num_workers=15,
        batch_size=8,
        pin_memory=False,
    ):
        """
        Initialize the LightningDataModule subclass for handling nodule data.

        Args:
            csv_path (str): Path to the CSV file containing the dataset information. Default is "/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/data/dataset_tcia.csv".
            num_workers (int): Number of workers to use for data loading. Default is 15.
            batch_size (int): Batch size for training, validation, and testing dataloaders. Default is 8.
            pin_memory (bool): Whether to pin memory for faster data transfer. Default is False.
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
        Method to set up the data module.

        Args:
            stage (str, optional): The stage being set up. Defaults to None.
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

        self.train_dataset = NoduleDataset(train_df, transform=self.transform)
        self.val_dataset = NoduleDataset(val_df, transform=self.transform)
        self.test_dataset = NoduleDataset(test_df, transform=self.transform)

    def train_dataloader(self):
        """
        Method to get the training dataloader.

        Returns:
            torch.utils.data.DataLoader: The training dataloader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Method to get the validation dataloader.

        Returns:
            torch.utils.data.DataLoader: The validation dataloader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Method to get the testing dataloader.

        Returns:
            torch.utils.data.DataLoader: The testing dataloader.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        Args:
            stage (str, optional): The stage being torn down. Either "fit", "validate", "test", or "predict". Defaults to None.
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
        """
        pass


def get_resnet3d():
    """
    Returns a 3D ResNet model for lung cancer classification.

    The function loads a pre-trained ResNet-50 model and updates its state dictionary
    with the weights from a pre-trained model file. The updated model is then returned.

    Returns:
        model_3d (torch.nn.Module): The 3D ResNet model for lung cancer classification.
    """
    model_3d = resnet50(sample_input_D=24, sample_input_H=48, sample_input_W=48, num_seg_classes=1)

    net_dict = model_3d.state_dict()

    pretrain = torch.load(
        "/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/pretrained_weights/resnet_50.pth",
        map_location=torch.device("cpu") if not torch.cuda.is_available() else "cuda",
    )
    pretrain_dict = {k: v for k, v in pretrain["state_dict"].items() if k in net_dict.keys()}

    net_dict.update(pretrain_dict)
    model_3d.load_state_dict(net_dict)

    return model_3d


class NodulesOnlyModule(LightningModule):
    """
    A LightningModule for training a model.

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """
        Initialize a `MNISTLitModule`.

        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: A boolean indicating whether to compile the model.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # BCEWithLogitsLoss but it penalizes more the false negatives
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))

        self.encoder = get_resnet3d()
        self.fc1 = torch.nn.Linear(576 * 3, 512)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(512, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.baseline = torch.nn.Parameter(torch.tensor([-30.0]), requires_grad=True).float()

        self.train_acc = BinaryAccuracy()

        self.val_acc = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_auroc = BinaryAUROC()
        self.val_specificity = BinarySpecificity()

        self.test_acc = BinaryAccuracy()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()
        self.test_auroc = BinaryAUROC()
        self.test_specificity = BinarySpecificity()
        self.test_stat_scores = BinaryStatScores()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, nod1: torch.Tensor, nod2: torch.Tensor, nod3: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param nod1: A tensor representing the first nodule.
        :param nod2: A tensor representing the second nodule.
        :param nod3: A tensor representing the third nodule.
        :return: A tensor of logits.
        """
        enc_nod1 = self.encoder(nod1)

        # flatten the encoded nodules
        enc_nod1 = enc_nod1.view(enc_nod1.size(0), -1)

        enc_nod2 = self.encoder(nod2)
        enc_nod2 = enc_nod2.view(enc_nod2.size(0), -1)

        enc_nod3 = self.encoder(nod3)
        enc_nod3 = enc_nod3.view(enc_nod3.size(0), -1)

        # concat the encoded nodules
        enc_nod = torch.cat((enc_nod1, enc_nod2, enc_nod3), dim=1)

        hidden = self.relu(self.fc1(enc_nod))
        y_hat = self.fc2(hidden)

        return y_hat

    def on_train_start(self) -> None:
        """
        Lightning hook that is called when training begins.
        """
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        # self.val_auroc.reset()
        self.val_specificity.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # nod1, nod2, nod3, y = batch
        # preds = self.forward(nod1, nod2, nod3)

        # iterate over the batch of predictions and targets
        # for i in range(preds.shape[0]):
        #     if y[i] == 1:
        #         # generate a random integer between 1 and 100
        #         random_int = np.random.randint(1, 101)
        #         random_int = str(random_int)

        #         if preds[i] != y[i]:
        #             # save nod1[i], nod2[i], nod3[i] to
        #             # "/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions"
        #             np.save(f"/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions/wrong_{random_int}_nod1_{i}.npy", nod1[i].detach().cpu().numpy())
        #             np.save(f"/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions/wrong_{random_int}_nod2_{i}.npy", nod2[i].detach().cpu().numpy())
        #             np.save(f"/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions/wrong_{random_int}_nod3_{i}.npy", nod3[i].detach().cpu().numpy())

        #         else:
        #             # save nod1[i], nod2[i], nod3[i] to
        #             # "/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions"
        #             np.save(f"/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions/correct_{random_int}_nod1_{i}.npy", nod1[i].detach().cpu().numpy())
        #             np.save(f"/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions/correct_{random_int}_nod2_{i}.npy", nod2[i].detach().cpu().numpy())
        #             np.save(f"/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions/correct_{random_int}_nod3_{i}.npy", nod3[i].detach().cpu().numpy())

        nod1, nod2, nod3, y = batch
        preds = self.forward(nod1, nod2, nod3)
        loss = self.criterion(preds, y)
        preds = self.sigmoid(preds)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """
        Lightning hook that is called when a training epoch ends.
        """
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data containing the input tensor of images and target labels.
            batch_idx (int): The index of the current batch.

        Returns:
            None
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Lightning hook that is called when a validation epoch ends.
        Computes and logs various validation metrics such as accuracy, precision, recall, F1 score, and specificity.
        """
        acc = self.val_acc.compute()  # get current val acc

        self.val_acc_best(acc)  # update best so far val acc

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/precision", self.val_precision.compute(), sync_dist=True, prog_bar=True)
        self.log("val/recall", self.val_recall.compute(), sync_dist=True, prog_bar=True)
        self.log("val/f1", self.val_f1.compute(), sync_dist=True, prog_bar=True)
        # self.log("val/auroc", self.val_auroc.compute(), sync_dist=True, prog_bar=True)
        self.log("val/specificity", self.val_specificity.compute(), sync_dist=True, prog_bar=True)

        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        # self.val_auroc.reset()
        self.val_specificity.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        nod1, nod2, nod3, y = batch
        preds = self.forward(nod1, nod2, nod3)

        # iterate over the batch of predictions and targets
        for i in range(preds.shape[0]):
            if y[i] == 1:
                # generate a random integer between 1 and 100
                random_int = np.random.randint(1, 101)
                random_int = str(random_int)

                if preds[i] > 0.5:
                    # save nod1[i], nod2[i], nod3[i] to
                    # "/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions"
                    np.save(
                        f"/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions/wrong_{random_int}_nod1_{i}.npy",
                        nod1[i].detach().cpu().numpy(),
                    )
                    np.save(
                        f"/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions/wrong_{random_int}_nod2_{i}.npy",
                        nod2[i].detach().cpu().numpy(),
                    )
                    np.save(
                        f"/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions/wrong_{random_int}_nod3_{i}.npy",
                        nod3[i].detach().cpu().numpy(),
                    )

                else:
                    # save nod1[i], nod2[i], nod3[i] to
                    # "/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions"
                    np.save(
                        f"/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions/correct_{random_int}_nod1_{i}.npy",
                        nod1[i].detach().cpu().numpy(),
                    )
                    np.save(
                        f"/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions/correct_{random_int}_nod2_{i}.npy",
                        nod2[i].detach().cpu().numpy(),
                    )
                    np.save(
                        f"/local_ssd/practical_wise24/lung_cancer/adlm-lung-cancer/logs/images/predictions/correct_{random_int}_nod3_{i}.npy",
                        nod3[i].detach().cpu().numpy(),
                    )

        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)

        self.test_precision.update(preds, targets)
        self.test_recall.update(preds, targets)
        self.test_f1.update(preds, targets)
        # self.test_auroc.update(preds, targets)
        self.test_specificity.update(preds, targets)
        self.test_stat_scores.update(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision.compute(), sync_dist=True, prog_bar=True)
        self.log("test/recall", self.test_recall.compute(), sync_dist=True, prog_bar=True)
        self.log("test/f1", self.test_f1.compute(), sync_dist=True, prog_bar=True)
        # self.log("test/auroc", self.test_auroc.compute(), sync_dist=True, prog_bar=True)

        try:
            print(f"test/test_stat_scores: {self.test_stat_scores.compute()}")

        except Exception as e:
            print(f"test/test_stat_scores: {repr(e)}")

    def on_test_epoch_end(self) -> None:
        """
        Lightning hook that is called when a test epoch ends.
        """
        try:
            print(f"test/test_stat_scores: {self.test_stat_scores.compute()}")

        except:
            pass

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer object.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    # data
    dm = NoduleDataModule()

    # model
    model = NodulesOnlyModule(
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        compile=True,
    )

    wandb.init(project="lightning-hydra-template", entity="adlm-lung-cancer")

    # Log train and test metrics
    wandb_logger = pl.loggers.WandbLogger()

    # training
    trainer = pl.Trainer(
        max_epochs=20,
        logger=wandb_logger,
    )

    trainer.fit(model, dm)

    # testing
    trainer.test(model, datamodule=dm)
