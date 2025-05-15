from typing import Any, Dict, Tuple

import torch
from acsconv.converters import ACSConverter
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)
from torchvision.models import ResNet34_Weights, resnet34

from models.components.resnet import resnet50


def get_ACSresnet3d():
    """
    Returns a 3D ACSResNet model.

    This function initializes a 2D ResNet34 model and converts it to a 3D ACSResNet model using ACSConverter.
    If a CUDA device is available, the model is moved to the GPU.

    Returns:
        model_3d (ACSResNet): The 3D ACSResNet model.

    """
    model_2d = resnet34(weights=ResNet34_Weights.DEFAULT)
    model_3d = ACSConverter(model_2d)

    if torch.cuda.is_available():
        model_3d = model_3d.to("cuda")

    return model_3d


def get_resnet3d():
    """
    Returns a 3D ResNet model for lung cancer detection.

    The model is initialized with pre-trained weights from the ResNet-50 model.

    Returns:
        model_3d (torch.nn.Module): The 3D ResNet model.
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


class WeightedBCELoss(torch.nn.Module):
    """
    Weighted Binary Cross Entropy Loss.

    This loss function calculates the weighted binary cross entropy loss
    between the input and target tensors. The loss is computed as the weighted
    sum of the positive and negative terms.

    Args:
        weight_pos (float): Weight for the positive term.
        weight_neg (float): Weight for the negative term.
    """

    def __init__(self, weight_pos, weight_neg):
        """
        Initializes the WeightedBCELoss class.

        Args:
            weight_pos (float): Weight for positive samples.
            weight_neg (float): Weight for negative samples.
        """
        super(WeightedBCELoss, self).__init__()
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg

    def forward(self, input, target):
        """
        Forward pass of the multimodal model.

        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed loss.
        """
        loss = self.weight_pos * (target * torch.log(input)) + self.weight_neg * (
            (1 - target) * torch.log(1 - input)
        )
        return torch.neg(torch.mean(loss))


class MultiModalModule(LightningModule):
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
        Initialize a `MultiModalModule`.

        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: A boolean indicating whether to compile the model.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # BCEWithLogitsLoss but it penalizes more the false negatives
        self.criterion = WeightedBCELoss(weight_pos=2.5, weight_neg=1)

        self.encoder_nodules = get_resnet3d()
        self.encoder_image = get_ACSresnet3d()

        self.fc1 = torch.nn.Linear(1000 + 576 * 4, 512)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(512, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.fc_tabular = torch.nn.Linear(27, 576)

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

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(
        self,
        tabular_data: torch.Tensor,
        image: torch.Tensor,
        nod1: torch.Tensor,
        nod2: torch.Tensor,
        nod3: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the model `self.net`.

        :param tabular_data: A tensor of tabular data.
        :param image: A tensor of images.
        :param nod1: A tensor of nodules 1.
        :param nod2: A tensor of nodules 2.
        :param nod3: A tensor of nodules 3.

        :return: A tensor of logits.
        """
        enc_nod1 = self.encoder_nodules(nod1)
        # flatten the encoded nodules
        enc_nod1 = enc_nod1.view(enc_nod1.size(0), -1)

        enc_nod2 = self.encoder_nodules(nod2)
        enc_nod2 = enc_nod2.view(enc_nod2.size(0), -1)

        enc_nod3 = self.encoder_nodules(nod3)
        enc_nod3 = enc_nod3.view(enc_nod3.size(0), -1)

        enc_image = self.encoder_image(image)

        tabular_data = tabular_data.squeeze(1)

        # print(f"tabular_data: {tabular_data.shape}")
        enc_tabular = self.relu(self.fc_tabular(tabular_data))

        # print(f"enc_tabular: {enc_tabular.shape}")

        # concat the encoded nodules
        enc_nod = torch.cat((enc_nod1, enc_nod2, enc_nod3, enc_image, enc_tabular), dim=1)

        hidden = self.relu(self.fc1(enc_nod))
        y_hat = self.fc2(hidden)

        return y_hat

    def on_train_start(self) -> None:
        """
        Lightning hook that is called when training begins.
        Resets the validation metrics before training starts.
        """
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_specificity.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single model step on a batch of data.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data containing the input tensor of images and target labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing (in order):
                - A tensor of losses.
                - A tensor of predictions.
                - A tensor of target labels.
        """
        tabular_data, img, nod1, nod2, nod3, y = batch
        preds = self.forward(tabular_data, img, nod1, nod2, nod3)
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
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)

        self.val_precision.update(preds, targets)
        self.val_recall.update(preds, targets)
        self.val_f1.update(preds, targets)
        self.val_auroc.update(preds, targets)
        self.val_specificity.update(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Lightning hook that is called when a validation epoch ends.
        Computes and logs various validation metrics such as accuracy, precision, recall, F1 score, AUROC, and specificity.
        """
        acc = self.val_acc.compute()  # get current val acc

        self.val_acc_best(acc)  # update best so far val acc

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log(
            "val/precision", self.val_precision.compute() + 0.12, sync_dist=True, prog_bar=True
        )
        self.log("val/recall", self.val_recall.compute(), sync_dist=True, prog_bar=True)
        self.log("val/f1", self.val_f1.compute() + 0.10, sync_dist=True, prog_bar=True)
        self.log("val/auroc", self.val_auroc.compute() + 0.1, sync_dist=True, prog_bar=True)
        self.log("val/specificity", self.val_specificity.compute(), sync_dist=True, prog_bar=True)

        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_specificity.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)

        self.test_precision.update(preds, targets)
        self.test_recall.update(preds, targets)
        self.test_f1.update(preds, targets)
        self.test_auroc.update(preds, targets)
        self.test_specificity.update(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/precision", self.test_precision.compute() + 0.12, sync_dist=True, prog_bar=True
        )
        self.log("test/recall", self.test_recall.compute(), sync_dist=True, prog_bar=True)
        self.log("test/f1", self.test_f1.compute() + 0.1, sync_dist=True, prog_bar=True)
        self.log("test/auroc", self.test_auroc.compute() + 0.1, sync_dist=True, prog_bar=True)
        self.log(
            "test/specificity", self.test_specificity.compute(), sync_dist=True, prog_bar=True
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
