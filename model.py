# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import functional
import math
from collections import OrderedDict


class LinearBottleneck(nn.Module):

    def __init__(self, input_channels, out_channels, expansion, stride=1, activation=nn.ReLU):
        super(LinearBottleneck, self).__init__()
        self.expansion_channels = input_channels * expansion

        self.conv1 = nn.Conv2d(input_channels, self.expansion_channels, stride=1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.expansion_channels)

        self.depth_conv2 = nn.Conv2d(self.expansion_channels, self.expansion_channels, stride=stride, kernel_size=3,
                                     groups=self.expansion_channels, padding=1)
        self.bn2 = nn.BatchNorm2d(self.expansion_channels)

        self.conv3 = nn.Conv2d(self.expansion_channels, out_channels, stride=1, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.activation = activation(inplace=True)  # inplace=True
        self.stride = stride
        self.input_channels = input_channels
        self.out_channels = out_channels

    def forward(self, input):
        residual = input

        out = self.conv1(input)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.depth_conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and self.input_channels == self.out_channels:
            out += residual
        return out


class MobileNetV2(nn.Module):

    def __init__(self, input_channels=3, nums_class=136, activation=nn.ReLU):
        super(MobileNetV2, self).__init__()
        self.coefficient = 0.50
        self.num_of_channels = [int(32 * self.coefficient), int(48 * self.coefficient), int(64 * self.coefficient),
                                int(64 * self.coefficient)]
        self.conv1 = nn.Conv2d(input_channels, self.num_of_channels[0], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_of_channels[0])

        self.stage0 = self.make_stage(self.num_of_channels[0], self.num_of_channels[1], stride=2, stage=0, times=2,
                                      expansion=2, activation=activation)

        self.stage1 = self.make_stage(self.num_of_channels[1], self.num_of_channels[2], stride=2, stage=1, times=2,
                                      expansion=2, activation=activation)

        self.conv2 = nn.Conv2d(self.num_of_channels[2], self.num_of_channels[3], kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_of_channels[3])

        self.activation = activation(inplace=True)

        self.in_features = 2 * 2 * self.num_of_channels[3]
        self.fc = nn.Linear(in_features=self.in_features, out_features=nums_class)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def make_stage(self, input_channels, out_channels, stride, stage, times, expansion, activation=nn.ReLU):
        modules = OrderedDict()
        stage_name = 'LinearBottleneck{}'.format(stage)

        module = LinearBottleneck(input_channels, out_channels, expansion=2,
                                  stride=stride, activation=activation)
        modules[stage_name+'_0'] = module

        for i in range(times - 1):
            module = LinearBottleneck(out_channels, out_channels, expansion=expansion, stride=1,
                                      activation=activation)
            module_name = stage_name+'_{}'.format(i+1)
            modules[module_name] = module

        return nn.Sequential(modules)

    def forward(self, input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.stage0(out)

        out = self.stage1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = out.view(out.size(0), -1)

        labels = self.fc(out)
        return labels


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super(BlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        if stride > 1:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=5, stride=stride, padding=2,
                      groups=in_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        out = (branch1+self.shortcut(x)) if self.use_pool else (branch1+x)
        return self.relu(out)


class DoubleBlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super(DoubleBlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        if stride > 1:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=stride,
                      padding=2, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=5, stride=1, padding=2,
                      groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        out = (branch1 + self.shortcut(x)) if self.use_pool else (branch1 + x)
        return self.relu(out)


class BlazeLandMark(nn.Module):
    def __init__(self, nums_class=136):
        super(BlazeLandMark, self).__init__()

        # self.firstconv = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(24),
        #     nn.ReLU(inplace=True),
        #
        #     # nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=2, padding=1),
        #     # nn.BatchNorm2d(24),
        #     # nn.ReLU(inplace=True),
        # )
        #
        # self.blazeBlock = nn.Sequential(
        #     BlazeBlock(in_channels=24, out_channels=24),
        #     BlazeBlock(in_channels=24, out_channels=24),
        #     BlazeBlock(in_channels=24, out_channels=48, stride=2),
        #     BlazeBlock(in_channels=48, out_channels=48),
        #     BlazeBlock(in_channels=48, out_channels=48),
        # )
        #
        # self.doubleBlazeBlock1 = nn.Sequential(
        #     DoubleBlazeBlock(in_channels=48, out_channels=96, mid_channels=24, stride=2),
        #     DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24),
        #     DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24))
        #
        # self.doubleBlazeBlock2 = nn.Sequential(
        #     DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24, stride=2),
        #     DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24),
        #     DoubleBlazeBlock(in_channels=96, out_channels=96, mid_channels=24),
        # )

        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
        )

        self.blazeBlock = nn.Sequential(
            BlazeBlock(in_channels=12, out_channels=12),
            BlazeBlock(in_channels=12, out_channels=12),
            BlazeBlock(in_channels=12, out_channels=24, stride=2),
            BlazeBlock(in_channels=24, out_channels=24),
            BlazeBlock(in_channels=24, out_channels=24),
        )

        self.doubleBlazeBlock1 = nn.Sequential(
            DoubleBlazeBlock(in_channels=24, out_channels=48, mid_channels=12, stride=2),
            DoubleBlazeBlock(in_channels=48, out_channels=48, mid_channels=12),
            DoubleBlazeBlock(in_channels=48, out_channels=48, mid_channels=12))

        self.doubleBlazeBlock2 = nn.Sequential(
            DoubleBlazeBlock(in_channels=48, out_channels=48, mid_channels=12, stride=2),
            DoubleBlazeBlock(in_channels=48, out_channels=48, mid_channels=12),
            DoubleBlazeBlock(in_channels=48, out_channels=48, mid_channels=12),
        )

        # self.secondconv = nn.Sequential(
        #     nn.Conv2d(in_channels=192, out_channels=1280, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(1280),
        #     nn.ReLU(inplace=True),
        # )

        # self.in_features = 96 + 96 + 128

        self.in_features = 48

        self.fc = nn.Linear(in_features=self.in_features, out_features=nums_class)

        self.init_params()

    # def initialize(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        fisrt_out = self.firstconv(input)

        block_out1 = self.blazeBlock(fisrt_out)

        block_out2 = self.doubleBlazeBlock1(block_out1)

        block_out3 = self.doubleBlazeBlock2(block_out2)

        features = functional.adaptive_avg_pool2d(block_out3, 1).squeeze(-1).squeeze(-1)

        assert features.size(1) == self.in_features
        pre_labels = self.fc(features)
        
        return pre_labels


class MBConvBlock(nn.Module):

    def __init__(self):
        super(MBConvBlock, self).__init__()


class EfficientNet(nn.Module):
    def __init__(self, nums_class=136):
        super(EfficientNet, self).__init__()

        self.firstconv = nn.Sequential(

        )

        self.in_features = 1280

        self.fc = nn.Linear(in_features=self.in_features, out_features=nums_class)

        self.init_params()

    # def initialize(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, input):
        return input

    def forward(self, input):

        features = self.extract_features(input)

        features = functional.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)

        assert features.size(1) == self.in_features
        pre_landmarks = self.fc(features)

        return pre_landmarks



class ONet(nn.Module):

    def __init__(self, nums_class=2):
        super(ONet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(128*5*5, 256)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(256, nums_class)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        # print("llllllllllll", x.size())
        x = self.fc1(x)

        x = self.relu5(x)
        labels = self.fc2(x)

        return labels


class PNet(nn.Module):
    '''PNet'''

    def __init__(self, nums_class=2):
        super(PNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

        self.fc1 = nn.Linear(32 *7 *7, 64)
        self.fc2 = nn.Linear(64, nums_class)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        labels = self.fc2(x)
        return labels


class RNet(nn.Module):
    ''' RNet '''

    def __init__(self, is_train=False, use_cuda=True):
        super(RNet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.PReLU()

        )
        # this is little different from MTCNN paper, cause in pytroch, pooliing is calculated by floor()
        self.conv4 = nn.Linear(64 * 2 * 2, 128)
        self.prelu4 = nn.PReLU()
        # face calssification
        self.conv5_1 = nn.Linear(128, 1)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)
        # lanbmark localization
        self.conv5_3 = nn.Linear(128, 10)
        # weight initiation weih xavier
        self.apply(weights_init)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        # x = x.view(-1, x.size(0))
        x = x.view(-1, 64 * 2 * 2)
        x = self.conv4(x)
        x = self.prelu4(x)
        # detection
        det = torch.sigmoid(self.conv5_1(x))
        box = self.conv5_2(x)

        if self.is_train is True:
            return det, box

        return det, box




