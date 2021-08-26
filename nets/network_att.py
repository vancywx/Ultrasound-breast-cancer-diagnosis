"""Feature encoder for ATDA.

it's called as `shared network` in the paper.
"""

import torch
from torch import nn
import torchvision
from attention import PAM_Module, CAM_Module


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4  # in_channels=512
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

#         self.conv6 = nn.Sequential(nn.Dropout2d(
#             0.1, False), nn.Conv2d(512, out_channels, 1))
#         self.conv7 = nn.Sequential(nn.Dropout2d(
#             0.1, False), nn.Conv2d(512, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(
            0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
#         sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
#         sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)
        return sasc_output

#         output = [sasc_output]
#         output.append(sa_output)
#         output.append(sc_output)
#         return tuple(output)


class ResNet_DA(nn.Module):

    def __init__(self, layers=18):
        """Init encoder."""
        super(ResNet_DA, self).__init__()

        self.restored = False
        if layers == 18:
            pretrained_model = torchvision.models.resnet18(pretrained=True)
        elif layers == 34:
            pretrained_model = torchvision.models.resnet34(pretrained=True)
        elif layers == 50:
            pretrained_model = torchvision.models.resnet50(pretrained=True)
        elif layers == 101:
            pretrained_model = torchvision.models.resnet101(pretrained=True)
        elif layers == 152:
            pretrained_model = torchvision.models.resnet152(pretrained=True)
        else:
            print('No such network exist...')
        mod = list(pretrained_model.children())
        mod.pop()
        mod.pop()
        self.head = DANetHead(512, 512, nn.BatchNorm2d)

        self.GAP = nn.AvgPool2d(9, 15)
        self.linear = nn.Linear(512, 2)

        self.encoder = nn.Sequential(*mod)

    def forward(self, x):
        """Forward encoder."""
        # x = expand_single_channel(x)
        out = self.encoder(x)
        out = self.head(out)
        out = self.GAP(out)
        out = out.view(-1, 512)
        out = self.linear(out)

        return out


class DenseNet_DA(nn.Module):

    def __init__(self, layers=121):
        """Init encoder."""
        super(DenseNet_DA, self).__init__()

        self.restored = False
        if layers == 121:
            pretrained_model = torchvision.models.densenet121(pretrained=True)
        elif layers == 161:
            pretrained_model = torchvision.models.densenet161(pretrained=True)
        elif layers == 201:
            pretrained_model = torchvision.models.densenet201(retrained=True)
        else:
            print('No such network exist...')
        mod = list(pretrained_model.children())
        mod.pop()
        self.head = DANetHead(1024, 512, nn.BatchNorm2d)

        self.GAP = nn.AvgPool2d(9, 15)
        self.linear = nn.Linear(512, 2)

        self.encoder = nn.Sequential(*mod)

    def forward(self, x):
        """Forward encoder."""
        # x = expand_single_channel(x)
        out = self.encoder(x)
        out = self.head(out)
        out = self.GAP(out)
        out = out.view(-1, 512)
        out = self.linear(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        """Init encoder."""
        super(ResNet, self).__init__()

        self.restored = False

        pretrained_model = torchvision.models.resnet18(pretrained=True)
        mod = list(pretrained_model.children())
        mod.pop()
        mod.pop()
        self.GAP = nn.AvgPool2d(7, 32)
        self.linear = nn.Linear(512, 2)

        self.encoder = nn.Sequential(*mod)

    def forward(self, x):
        """Forward encoder."""
        # x = expand_single_channel(x)
        out = self.encoder(x)
        out = self.GAP(out)
        out = out.view(-1, 512)
        out = self.linear(out)

        return out


class DenseNet121(nn.Module):
    """Feature encoder class for MNIST -> MNIST-M experiment in ATDA."""

    def __init__(self):
        """Init encoder."""
        super(DenseNet121, self).__init__()

        self.restored = False

        pretrained_model = torchvision.models.densenet121(pretrained=True)
        mod = list(pretrained_model.children())
        mod.pop()
        self.GAP = nn.AvgPool2d(9, 15)
        self.linear = nn.Linear(1024, 2)

        self.encoder = nn.Sequential(*mod)

    def forward(self, x):
        """Forward encoder."""
        # x = expand_single_channel(x)
        out = self.encoder(x)
        out = self.GAP(out)
        out = out.view(-1, 1024)
        out = self.linear(out)

        return out
