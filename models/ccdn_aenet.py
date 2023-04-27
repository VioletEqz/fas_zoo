from typing import List
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Conv2d_Hori_Veri_Cross(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_Hori_Veri_Cross, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(
            1, 5), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):

        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        tensor_zeros = torch.FloatTensor(C_out, C_in, 1).fill_(0)
        conv_weight = torch.cat((tensor_zeros, self.conv.weight[:, :, :, 0], tensor_zeros, self.conv.weight[:, :, :, 1],
                                self.conv.weight[:, :, :, 2], self.conv.weight[:, :, :, 3], tensor_zeros, self.conv.weight[:, :, :, 4], tensor_zeros), 2)
        conv_weight = conv_weight.contiguous().view(C_out, C_in, 3, 3)

        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias,
                              stride=self.conv.stride, padding=self.conv.padding)

        # if torch.abs(torch.tensor[self.theta] - 0.0) < 1e-8:
        if np.abs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return Conv2d_Hori_Veri_Cross(in_planes, out_planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = Conv2d_Hori_Veri_Cross(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.act = nn.Hardswish()
        # self.conv2 = conv3x3(planes, planes)
        self.conv2 = Conv2d_Hori_Veri_Cross(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out

# AENet_C,S,G is based on ResNet-18


class AENet(nn.Module):

    def __init__(
            self,
            block=BasicBlock,
            widths: List = [32, 64, 128, 256],
            layers: List = [2, 2, 2, 2],
            reduced: bool = False,
            theta=0.0,
            use_depth=True,
            use_ref=True,
    ):
        self.inplanes = widths[0]
        self.use_depth = use_depth
        self.use_ref = use_ref
        super(AENet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = Conv2d_Hori_Veri_Cross(3, 64, kernel_size=7, stride=2, padding=1,
                                            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.act = nn.ReLU(inplace=True)
        self.act = nn.Hardswish()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, widths[0], layers[0])
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2)
        if not reduced:
            self.layer4 = self._make_layer(
                block, widths[3], layers[3], stride=2)
            last_width = widths[3]
        else:
            last_width = widths[2]
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # Three classifiers of semantic informantion
        self.fc_live_attribute = nn.Linear(last_width * block.expansion, 40)
        self.fc_attack = nn.Linear(last_width * block.expansion, 11)
        self.fc_light = nn.Linear(last_width * block.expansion, 5)
        # One classifier of Live/Spoof information
        self.fc_live = nn.Linear(last_width * block.expansion, 2)

        # Two embedding modules of geometric information
        self.upsample14 = nn.Upsample((14, 14), mode='bilinear')
        # self.depth_final = nn.Conv2d(
        #     512, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.reflect_final = nn.Conv2d(
        #     512, 3, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_depth:
            self.depth_final = Conv2d_Hori_Veri_Cross(
                last_width, 1, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_ref:
            self.reflect_final = Conv2d_Hori_Veri_Cross(
                last_width, 3, kernel_size=3, stride=1, padding=1, bias=False)
        # The ground truth of depth map and reflection map has been normalized[torchvision.transforms.ToTensor()]
        self.sigmoid = nn.Sigmoid()

        # initialization
        for m in self.modules():
            if isinstance(m, Conv2d_Hori_Veri_Cross):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                n = m.kernel_size * m.kernel_size * m.out_channels
                m = nn.init.kaiming_normal_(
                    m.conv.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        print(x.shape)
        if hasattr(self, 'layer4'):
            x = self.layer4(x)
            print(x.shape)
        else:
            # If we don't have layer4, we have to reduce the size of the feature map
            # currently at 14x14, must be 7x7
            x = self.avgpool(x)
            # now it's at 2x2, reduce it further to 1x1
            x = self.maxpool(x)
            print(x.shape)
        depth_map = None
        if self.use_depth:
            depth_map = self.depth_final(x)
            depth_map = self.sigmoid(depth_map)
            depth_map = self.upsample14(depth_map)

        reflect_map = None
        if self.use_ref:
            reflect_map = self.reflect_final(x)
            reflect_map = self.sigmoid(reflect_map)
            reflect_map = self.upsample14(reflect_map)

        x = self.avgpool(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)

        x_live_attribute = self.fc_live_attribute(x)
        x_attack = self.fc_attack(x)
        x_light = self.fc_light(x)
        x_live = self.fc_live(x)

        # return all the output
        return depth_map, reflect_map, x_live_attribute, x_attack, x_light, x_live


if __name__ == "__main__":
    # Randomly generate a 3-channel image with a size of 224*224
    input = torch.rand(1, 3, 224, 224)
    model = AENet(use_depth=False, use_ref=False, reduced=False)
    # model = AENet(use_depth=False, use_ref=False, reduced=True)
    model.eval()
    output = model(input)
    # print(output)
    print('wewew')
