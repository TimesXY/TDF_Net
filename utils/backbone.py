import torch
import torchvision
import torch.nn.functional as F

from torch import nn


class SharedBackBone(nn.Module):
    def __init__(self, pretrain=True, depth=-3):
        super().__init__()

        # Establishment of a backbone network
        self.backbone = torchvision.models.resnet50()

        if pretrain:
            model_weight_path = r"D:/PycharmScript/Models/resnet50.pth"
            self.backbone.load_state_dict(torch.load(model_weight_path), strict=False)
            print('Pre-training weight loading complete')

        # Remove backbone's layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:depth])

    def forward(self, x1, x2, x3):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x3 = self.backbone(x3)

        return x1, x2, x3


class SharedBackBoneFPN(nn.Module):
    def __init__(self, pretrain=True, depth=None):
        super().__init__()

        # Establishment of a backbone network
        self.backbone = torchvision.models.resnet50()

        # Load pre-training weights
        if depth is None:
            depth = [-5, -4, -3]

        if pretrain:
            model_weight_path = r"D:/PycharmScript/Models/resnet50.pth"
            self.backbone.load_state_dict(torch.load(model_weight_path), strict=False)
            print('Pre-training weight loading complete')

        # Remove backbone's layers
        self.backbone_1 = nn.Sequential(*list(self.backbone.children())[:depth[0]])
        self.backbone_2 = nn.Sequential(*list(self.backbone.children())[:depth[1]])
        self.backbone_3 = nn.Sequential(*list(self.backbone.children())[:depth[2]])

        # Use convolutional layers to halve the same size of the feature map
        self.conv_d = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.conv_s1 = nn.Conv2d(1024, 512, kernel_size=(1, 1))
        self.conv_s2 = nn.Conv2d(1024, 512, kernel_size=(1, 1))
        self.conv_s3 = nn.Conv2d(1024, 512, kernel_size=(1, 1))

    def forward(self, x1, x2, x3):
        x1_1 = self.backbone_1(x1)
        x1_2 = self.backbone_1(x2)
        x1_3 = self.backbone_1(x3)

        x1_1 = self.conv_d(x1_1)
        x1_2 = self.conv_d(x1_2)
        x1_3 = self.conv_d(x1_3)

        x2_1 = self.backbone_2(x1)
        x2_2 = self.backbone_2(x2)
        x2_3 = self.backbone_2(x3)

        x3_1 = self.backbone_3(x1)
        x3_2 = self.backbone_3(x2)
        x3_3 = self.backbone_3(x3)

        x3_1 = F.interpolate(x3_1, scale_factor=2, mode='bilinear', align_corners=False)
        x3_2 = F.interpolate(x3_2, scale_factor=2, mode='bilinear', align_corners=False)
        x3_3 = F.interpolate(x3_3, scale_factor=2, mode='bilinear', align_corners=False)

        x1 = torch.cat((x1_1, x2_1), dim=1) + x3_1
        x2 = torch.cat((x1_2, x2_2), dim=1) + x3_2
        x3 = torch.cat((x1_3, x2_3), dim=1) + x3_3

        x1 = self.conv_s1(x1)
        x2 = self.conv_s2(x2)
        x3 = self.conv_s3(x3)

        return x1, x2, x3


class SharedTransformer(nn.Module):
    def __init__(self, pretrain=True, depth=-2):
        super().__init__()

        # Establishment of a backbone network
        self.backbone = torchvision.models.vit_b_16()

        # Load pre-training weights
        if pretrain:
            model_weight_path = r"D:/PycharmScript/Models/vit_b_16.pth"
            self.backbone.load_state_dict(torch.load(model_weight_path), strict=False)
            print('Pre-training weight loading complete')

        # Remove backbone's layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:depth])

    def forward(self, x1, x2, x3):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x3 = self.backbone(x3)
        return x1, x2, x3


if __name__ == '__main__':
    # [batch_size, channel, height, width]
    A1 = torch.rand([32, 3, 224, 224])
    A2 = torch.rand([32, 3, 224, 224])
    A3 = torch.rand([32, 3, 224, 224])

    shared = SharedBackBone(pretrain=True, depth=-4)
    C1, C2, C3 = shared(A1, A2, A3)

    print(C1.shape)
    print(C2.shape)
    print(C3.shape)
