import torch
import torch.nn as nn

import torchvision.models as models
from torchvision.models import DenseNet201_Weights

from pytorch_grad_cam import GradCAM



class DenseNet201(nn.Module):
    def __init__(self, num_classes=2, quantize=False):
        super(DenseNet201, self).__init__()

        self.quantize = quantize

        self.model = models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # num_neurons = self.model.classifier.in_features
        # self.model.classifier = nn.Linear(num_neurons, num_classes)

        # from the source code of the MDPI paper
        self.model.classifier = nn.Sequential(
            # nn.AvgPool2d((1,1)),
            # nn.Flatten(),
            nn.Linear(1920, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        self.quant = torch.quantization.QuantStub() if quantize else None
        self.dequant = torch.quantization.DeQuantStub() if quantize else None

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)

        x = self.model(x)

        if self.quantize:
            x = self.dequant(x)
        return x
    

class DenseNet201ModifiedInput(nn.Module):
    def __init__(self, num_classes=2, in_channels=4):
        super(DenseNet201ModifiedInput, self).__init__()

        # Replace existing first convolution layer with one accepting 4 channels
        self.first_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3
        )

        # Rest of the code stays the same
        self.model = models.densenet201(pretrained=True)  # Use pre-trained weights

        # Remove the first layer from the pre-trained model
        self.model.features = nn.Sequential(*list(self.model.features.children())[1:])

        self.model.classifier = nn.Sequential(
            nn.Linear(1920, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.model(x)
        return x

    

# print the model
if __name__ == '__main__':
    model = DenseNet201()
    print(model)
    print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))