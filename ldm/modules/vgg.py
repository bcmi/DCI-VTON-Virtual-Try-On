import torch
from torch import nn
import torch.nn.functional as F


def vgg_preprocess(tensor, vgg_normal_correct=False):
    if vgg_normal_correct:
        tensor = (tensor + 1) / 2
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = torch.cat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    # tensor_bgr = tensor[:, [2, 1, 0], ...]
    tensor_bgr_ml = tensor_bgr - torch.Tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(
        1, 3, 1, 1
    )
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst


class VGG19_feature_color_torchversion(nn.Module):
    """
    NOTE: there is no need to pre-process the input
    input tensor should range in [0,1]
    """

    def __init__(self, pool="max", vgg_normal_correct=True, ic=3):
        super(VGG19_feature_color_torchversion, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct

        self.conv1_1 = nn.Conv2d(ic, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == "max":
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == "avg":
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys, preprocess=True):
        """
        NOTE: input tensor should range in [0,1]
        """
        out = {}
        if preprocess:
            x = vgg_preprocess(x, vgg_normal_correct=self.vgg_normal_correct)
        out["r11"] = F.relu(self.conv1_1(x))
        out["r12"] = F.relu(self.conv1_2(out["r11"]))
        out["p1"] = self.pool1(out["r12"])
        out["r21"] = F.relu(self.conv2_1(out["p1"]))
        out["r22"] = F.relu(self.conv2_2(out["r21"]))
        out["p2"] = self.pool2(out["r22"])
        out["r31"] = F.relu(self.conv3_1(out["p2"]))
        out["r32"] = F.relu(self.conv3_2(out["r31"]))
        out["r33"] = F.relu(self.conv3_3(out["r32"]))
        out["r34"] = F.relu(self.conv3_4(out["r33"]))
        out["p3"] = self.pool3(out["r34"])
        out["r41"] = F.relu(self.conv4_1(out["p3"]))
        out["r42"] = F.relu(self.conv4_2(out["r41"]))
        out["r43"] = F.relu(self.conv4_3(out["r42"]))
        out["r44"] = F.relu(self.conv4_4(out["r43"]))
        out["p4"] = self.pool4(out["r44"])
        out["r51"] = F.relu(self.conv5_1(out["p4"]))
        out["r52"] = F.relu(self.conv5_2(out["r51"]))
        out["r53"] = F.relu(self.conv5_3(out["r52"]))
        out["r54"] = F.relu(self.conv5_4(out["r53"]))
        out["p5"] = self.pool5(out["r54"])
        return [out[key] for key in out_keys]
