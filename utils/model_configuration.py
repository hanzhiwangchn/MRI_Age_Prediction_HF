import math
import torch.nn as nn
from transformers import ResNetForImageClassification, ResNetModel


# ------------------- PretTrained HF ResNet ---------------------
class ResNetModelHF(nn.Module):
    """
    PreTrained ResNet model without classification head + custom regression head
    """
    def __init__(self):
        super(ResNetModelHF, self).__init__()
        self.base_model = ResNetModel.from_pretrained('microsoft/resnet-50')
        self.custom_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.5),
            nn.Linear(2048 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values)
        # outputs return an HF module, so we extract the values from it
        outputs = self.custom_head(outputs[0])
        return outputs


class ResNetModelWithHeadHF(nn.Module):
    """
    PreTrained ResNet model with classification head + custom regression head
    """
    def __init__(self):
        super(ResNetModelWithHeadHF, self).__init__()
        self.base_model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')
        self.custom_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1000, 1),
        )

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values)
        # outputs return an HF module, so we extract the values from it
        outputs = self.custom_head(outputs[0])
        return outputs


# ------------------- Scratch ResNet ---------------------

class ResNetBlock(nn.Module):
    def __init__(self, block_number, input_size):
        super(ResNetBlock, self).__init__()

        layer_in = input_size if input_size is not None else 2 ** (block_number + 1)
        layer_out = 2 ** (block_number + 2)

        self.conv1 = nn.Conv3d(layer_in, layer_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(layer_out)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv3d(layer_out, layer_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(layer_out)
        self.shortcut = nn.Conv3d(layer_in, layer_out, kernel_size=1, stride=1, bias=False)
        self.act2 = nn.ELU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.act2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_size):
        super(ResNet, self).__init__()

        self.layer1 = self._make_block(1, input_size[0])
        self.layer2 = self._make_block(2)
        self.layer3 = self._make_block(3)
        self.layer4 = self._make_block(4)
        self.layer5 = self._make_block(5)

        d, h, w = ResNet._maxpool_output_size(input_size[1::], nb_layers=5)

        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.5),
            nn.Linear(128 * d * h * w, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    @staticmethod
    def _make_block(block_number, input_size=None):
        return nn.Sequential(
            ResNetBlock(block_number, input_size),
            nn.MaxPool3d(3, stride=2))

    @staticmethod
    def _maxpool_output_size(input_size, kernel_size=(3, 3, 3), stride=(2, 2, 2), nb_layers=1):
        d = math.floor((input_size[0] - kernel_size[0]) / stride[0] + 1)
        h = math.floor((input_size[1] - kernel_size[1]) / stride[1] + 1)
        w = math.floor((input_size[2] - kernel_size[2]) / stride[2] + 1)

        if nb_layers == 1:
            return d, h, w
        return ResNet._maxpool_output_size(input_size=(d, h, w), kernel_size=kernel_size,
                                           stride=stride, nb_layers=nb_layers - 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.fc(out)
        return out


# ------------------- ResNet for ABIDE----------------

class ResNetBlock_stride(nn.Module):
    def __init__(self, block_number, input_size):
        super(ResNetBlock_stride, self).__init__()

        layer_in = input_size if input_size is not None else 2 ** (block_number + 1)
        layer_out = 2 ** (block_number + 2) if block_number < 5 else 64

        self.conv1 = nn.Conv3d(layer_in, layer_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(layer_out)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv3d(layer_out, layer_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(layer_out)
        self.shortcut = nn.Conv3d(layer_in, layer_out, kernel_size=1, stride=2, bias=False)
        self.act2 = nn.ELU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.act2(out)
        return out


class ResNet_stride(nn.Module):
    def __init__(self, input_size):
        super(ResNet_stride, self).__init__()

        self.layer1 = self._make_block(1, input_size[0])
        self.layer2 = self._make_block(2)
        self.layer3 = self._make_block(3)
        self.layer4 = self._make_block(4)
        self.layer5 = self._make_block(5)

        self.avgpool = nn.AdaptiveAvgPool3d((2, 2, 2))

        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.5),
            nn.Linear(64 * 2 * 2 * 2, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    @staticmethod
    def _make_block(block_number, input_size=None):
        return nn.Sequential(
            ResNetBlock_stride(block_number, input_size))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    model = ResNetModelHF()
    input_size = (3, 224, 224)
    # input_size = (1, 96, 112, 96)
    # model = ResNet(input_size=input_size)
    # summary(model, input_size)
