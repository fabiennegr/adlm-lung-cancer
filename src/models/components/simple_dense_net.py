import lightning as L
import torch
from torch import nn
from torchvision.models import resnet34


class SimpleDenseNet(L.LightningModule):
    """ResNet34 model for lung cancer nodules risk prediciton on 2d slices"""

    def __init__(
        self,
        input_size: int = 784,
        output_size: int = 1,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param output_size: The number of output features of the final linear layer. Defaults to 1 to predict cancer or no cancer.
        """
        super().__init__()

        self.model = resnet34(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor. Already flattened.
        :return: A tensor of predictions.
        """

        return self.model(x)


if __name__ == "__main__":
    _ = SimpleDenseNet()
