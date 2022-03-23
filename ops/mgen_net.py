 # Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
from ops.base_module import *
# from grafting_ops.Gan_modules import *
import torchvision


class STDM(nn.Module):
    def __init__(self, resnet_model, resnet_model1, apha, belta):
        super(STDM, self).__init__()

        # implement conv1_5 and inflate weight
        self.conv1_temp = list(resnet_model1.children())[0]
        params = [x.clone() for x in self.conv1_temp.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * 2,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        self.conv1_5 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_5[0].weight.data = new_kernels

        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.fc = list(resnet_model.children())[8]
        self.apha = apha
        self.belta = belta

    def forward(self, x):
        x1, x2, x3 = x[:, 0:3, :, :], x[:, 3:6, :, :], x[:, 6:9, :, :]
        x_c5 = self.conv1_5(self.avg_diff(torch.cat([x2 - x1, x3 - x2], 1).view(-1, 6, x2.size()[2], x2.size()[3])))
        x_diff = self.maxpool_diff(1.0 / 1.0 * x_c5)

        temp_out_diff1 = x_diff

        temp_out_diff1 = F.interpolate(temp_out_diff1, (56, 56))

        return temp_out_diff1


class Mgen_NetP1(nn.Module):

    def __init__(self, num_segments, latent_dim=256):
        super(Mgen_NetP1, self).__init__()

        self.num_segments = num_segments
        self.latent_dim = latent_dim

        self.pad1_forward = ([0, 0, 0, 0, 0, 0, 0, 1])
        self.pad1_backward = ([0, 0, 0, 0, 0, 0, 1, 0])

        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)

        self.latent_init = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.encoder_ds = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.dense_fuse_1 = nn.Conv3d(in_channels=128 + 64, out_channels=128, kernel_size=(1, 1, 1))

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        self.dense_fuse_2 = nn.Conv3d(in_channels=256 + 128 + 64, out_channels=256, kernel_size=(1, 1, 1))

        self.hidden_conv = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.decoder_stage_1 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder_stage_2 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        # x: (B x T) x C x H x W
        x_uf = x.view((-1, self.num_segments) + x.size()[1:])
        _, fea_forward = x_uf.split([1, self.num_segments - 1], dim=1)  # B x (T - 1) x C x H x W
        fea_backward, _ = x_uf.split([self.num_segments - 1, 1], dim=1)  # B x (T - 1) x C x H x W

        fea_forward = F.pad(fea_forward, self.pad1_forward, mode="constant", value=0)  # B x T x C x H x W
        fea_forward = fea_forward - x_uf
        fea_forward = fea_forward.view((-1,) + x.size()[1:])  # (B x T) x C x H x W

        fea_backward = F.pad(fea_backward, self.pad1_backward, mode="constant", value=0)   # B x T x C x H x W
        fea_backward = x_uf - fea_backward
        fea_backward = fea_backward.view((-1,) + x.size()[1:])  # (B x T) x C x H x W

        x = torch.cat((fea_forward, fea_backward), dim=1)

        # x: (B x T) x C x H x W
        x = self.avg_diff(x)
        x = self.latent_init(x)

        nt, c, h, w = x.size()
        n = nt // self.num_segments

        x = x.view((n, self.num_segments, c, h, w)).transpose(1, 2)

        # x: B x C x T x H x W
        motion = self.encoder_stage1(x)
        motion = self.encoder_ds(motion)
        x_1 = self.encoder_ds(x)
        x = torch.cat((motion, x_1), dim=1)
        x = self.dense_fuse_1(x)

        motion = self.encoder_stage2(x)
        motion = self.encoder_ds(motion)
        x_2 = self.encoder_ds(x)
        x_1 = self.encoder_ds(x_1)
        x = torch.cat((motion, x_2, x_1), dim=1)
        x = self.dense_fuse_2(x)

        x = self.hidden_conv(x)

        x = F.interpolate(x, (self.num_segments, h // 2, w // 2))
        x = self.decoder_stage_1(x)
        x = F.interpolate(x, (self.num_segments, h, w))
        x = self.decoder_stage_2(x)

        x = x.transpose(1, 2).contiguous().view(nt, c, h, w)

        return x


class Mgen_NetP2(nn.Module):

    def __init__(self, resnet_model, resnet_model1):
        super(Mgen_NetP2, self).__init__()
        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)

        self.resnext_layer1 = nn.Sequential(*list(resnet_model1.children())[4])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.avg_diff = nn.AvgPool2d(kernel_size=2,stride=2)
        self.fc = list(resnet_model.children())[8]

        self.alpha = 0.5
        self.beta = 0.5

    def forward(self, x, motion):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # fusion layer1
        x = self.maxpool(x)

        x = self.alpha * x + self.beta * motion

        # fusion layer2
        x = self.layer1_bak(x)
        x_diff = self.resnext_layer1(motion)
        x_diff = F.interpolate(x_diff, x.size()[2:])

        x = self.alpha * x + self.beta * x_diff

        x = self.layer2_bak(x)
        x = self.layer3_bak(x)
        x = self.layer4_bak(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class Mgen_Net(nn.Module):

    def __init__(self, resnet_model, resnet_model1, num_segments):
        super(Mgen_Net, self).__init__()

        # if test:
        #     self.p1 = STDM(resnet_model, resnet_model1, apha=0.5, belta=0.5)
        # else:
        self.p1 = Mgen_NetP1(num_segments=num_segments)
        self.p2 = Mgen_NetP2(resnet_model, resnet_model1)

    def forward(self, x):
        with torch.no_grad():
            motion = self.p1(x)
        x = self.p2(x, motion)

        return x

def mgen_net(base_model=None, num_segments=8, step=1, pretrained=True, **kwargs):
    if("50" in base_model):
        resnet_model = fbresnet50(num_segments, pretrained)
        resnet_model1 = fbresnet50(num_segments, pretrained)
    else:
        resnet_model = fbresnet101(num_segments, pretrained)
        resnet_model1 = fbresnet101(num_segments, pretrained)

    if step == 1:
        model = Mgen_NetP1(num_segments)
    elif step == 2 or step == 3:
        model = Mgen_Net(resnet_model, resnet_model1, num_segments)
    else:
        raise ValueError("Grafting step incorrect.")

    return model

