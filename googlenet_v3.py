from torch.nn import Module, Sequential, LeakyReLU, Conv2d, BatchNorm2d, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Linear
import torch

class GoogLeNetV3(Module):
    def __init__(self, channels_in):
        super(GoogLeNetV3, self).__init__()
        self.in_block = Sequential(
            Conv2d_BN(channels_in, 32, 3, stride=2, padding=1),  # size /= 2
            Conv2d_BN(32, 32, 3, stride=1, padding=1),
            Conv2d_BN(32, 64, 3, stride=1, padding=1),
            MaxPool2d(3, stride=2, padding=1),  # size /= 2
            Conv2d_BN(64, 80, 1, stride=1, padding=0),
            Conv2d_BN(80, 192, 3, stride=1, padding=1),
            MaxPool2d(3, stride=2, padding=1)  # size /= 2
        )  # 192 channels
        self.mix_block = Sequential(
            InceptionA(192, 32),
            InceptionA(256, 64),
            InceptionA(288, 64),
            InceptionB(288),  # size /= 2
            InceptionC(768, 128),
            InceptionC(768, 160),
            InceptionC(768, 160),
            InceptionC(768, 192),
            InceptionD(768),  # size /= 2
            InceptionE(1280),
            InceptionE(2048)
        )  # 2048 channels
        self.out_block = Sequential(
            Conv2d_BN(2048, 1024, 1, stride=1, padding=0),
            AdaptiveAvgPool2d(1)
        )  # 1024 channels
        self.full_connect = Linear(1024, 1)
        
    def forward(self, x):
        x = self.in_block(x)
        x = self.mix_block(x)
        x = self.out_block(x)
        x = torch.flatten(x, 1)
        return self.full_connect(x)

class Conv2d_BN(Module):
    def __init__(self, channels_in, channels_out, kernel_size, padding, stride=1, acti=LeakyReLU(0.2, inplace=True)):
        super(Conv2d_BN, self).__init__()
        self.conv2d_bn = Sequential(
            Conv2d(channels_in, channels_out, kernel_size, stride, padding, bias=False),
            BatchNorm2d(channels_out),
            acti
        )

    def forward(self, x):
        return self.conv2d_bn(x)

class InceptionA(Module):
    def __init__(self, channels_in, pool_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = Conv2d_BN(channels_in, 64, 1, stride=1, padding=0)  # 64 channels
        self.branch5x5 = Sequential(
            Conv2d_BN(channels_in, 48, 1, stride=1, padding=0),
            Conv2d_BN(48, 64, 5, stride=1, padding=2)
        )  # 64 channels
        self.branch3x3dbl = Sequential(
            Conv2d_BN(channels_in, 64, 1, stride=1, padding=0),
            Conv2d_BN(64, 96, 3, stride=1, padding=1),
            Conv2d_BN(96, 96, 3, stride=1, padding=1)
        )  # 96 channels
        self.branch_pool = Sequential(
            AvgPool2d(3, stride=1, padding=1),
            Conv2d_BN(channels_in, pool_channels, 1, stride=1, padding=0)
        )  # pool_channels

    def forward(self, x):
        outputs = [self.branch1x1(x), self.branch5x5(x), self.branch3x3dbl(x), self.branch_pool(x)]
        # 64 + 64 + 96 + pool_channels
        return torch.cat(outputs, 1)

class InceptionB(Module):
    def __init__(self, channels_in):
        super(InceptionB, self).__init__()
        self.branch3x3 = Conv2d_BN(channels_in, 384, 3, stride=2, padding=1)  # 384 channels
        self.branch3x3dbl = Sequential(
            Conv2d_BN(channels_in, 64, 1, padding=0),
            Conv2d_BN(64, 96, 3, padding=1),
            Conv2d_BN(96, 96, 3, stride=2, padding=1)
        )  # 96 channels
        self.branch_pool = MaxPool2d(3, stride=2, padding=1)  # channels_in

    def forward(self, x):
        outputs = [self.branch3x3(x), self.branch3x3dbl(x), self.branch_pool(x)]
        # 384 + 96 + channels_in
        return torch.cat(outputs, 1)

class InceptionC(Module):
    def __init__(self, channels_in, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = Conv2d_BN(channels_in, 192, 1, stride=1, padding=0)  # 192 channels
        self.branch7x7 = Sequential(
            Conv2d_BN(channels_in, channels_7x7, 1, stride=1, padding=0),
            Conv2d_BN(channels_7x7, channels_7x7, (1, 7), stride=1, padding=(0, 3)),
            Conv2d_BN(channels_7x7, 192, (7, 1), stride=1, padding=(3, 0))
        )  # 192 channels
        self.branch7x7dbl = Sequential(
            Conv2d_BN(channels_in, channels_7x7, 1, stride=1, padding=0),
            Conv2d_BN(channels_7x7, channels_7x7, (7, 1), stride=1, padding=(3, 0)),
            Conv2d_BN(channels_7x7, channels_7x7, (1, 7), stride=1, padding=(0, 3)),
            Conv2d_BN(channels_7x7, channels_7x7, (7, 1), stride=1, padding=(3, 0)),
            Conv2d_BN(channels_7x7, 192, (1, 7), stride=1, padding=(0, 3))
        )  # 192 channels
        self.branch_pool = Sequential(
            AvgPool2d(3, stride=1, padding=1),
            Conv2d_BN(channels_in, 192, 1, stride=1, padding=0)
        )  # 192 channels
    
    def forward(self, x):
        outputs = [self.branch1x1(x), self.branch7x7(x), self.branch7x7dbl(x), self.branch_pool(x)]
        # 192 + 192 + 192 + 192 = 768 channels
        return torch.cat(outputs, 1)

class InceptionD(Module):
    def __init__(self, channels_in):
        super(InceptionD, self).__init__()
        self.branch3x3 = Sequential(
            Conv2d_BN(channels_in, 192, 1, stride=1, padding=0),
            Conv2d_BN(192, 320, 3, stride=2, padding=1)
        )  # 320 channels
        self.branch7x7x3 = Sequential(
            Conv2d_BN(channels_in, 192, 1, stride=1, padding=0),
            Conv2d_BN(192, 192, (1, 7), stride=1, padding=(0, 3)),
            Conv2d_BN(192, 192, (7, 1), stride=1, padding=(3, 0)),
            Conv2d_BN(192, 192, 3, stride=2, padding=1)
        )  # 192 chnnels
        self.branch_pool = MaxPool2d(3, stride=2, padding=1)  # channels_in

    def forward(self, x):
        outputs = [self.branch3x3(x), self.branch7x7x3(x), self.branch_pool(x)]
        # 320 + 192 + channels_in
        return torch.cat(outputs, 1)

class InceptionE(Module):
    def __init__(self, channels_in):
        super(InceptionE, self).__init__()
        self.branch1x1 = Conv2d_BN(channels_in, 320, 1, stride=1, padding=0)  # 320 channels

        self.branch3x3_1 = Conv2d_BN(channels_in, 384, 1, stride=1, padding=0)
        self.branch3x3_2a = Conv2d_BN(384, 384, (1, 3), stride=1, padding=(0, 1))
        self.branch3x3_2b = Conv2d_BN(384, 384, (3, 1), stride=1, padding=(1, 0))
        # 768 channels

        self.branch3x3dbl_1 = Sequential(
            Conv2d_BN(channels_in, 448, 1, stride=1, padding=0),
            Conv2d_BN(448, 384, 3, stride=1, padding=1)
        )
        self.branch3x3dbl_2a = Conv2d_BN(384, 384, (1, 3), stride=1, padding=(0, 1))
        self.branch3x3dbl_2b = Conv2d_BN(384, 384, (3, 1), stride=1, padding=(1, 0))
        # 768 channels
        
        self.branch_pool = Sequential(
            AvgPool2d(3, stride=1, padding=1),
            Conv2d_BN(channels_in, 192, 1, stride=1, padding=0)
        )  # 192 channels
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat([self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)], 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = torch.cat([self.branch3x3dbl_2a(branch3x3dbl), self.branch3x3dbl_2b(branch3x3dbl)], 1)

        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        # 320 + 768 + 768 + 192 = 2048 channels
        return torch.cat(outputs, 1)
