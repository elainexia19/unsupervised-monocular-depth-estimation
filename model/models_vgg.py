from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size - 1) / 2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return F.elu(x, inplace=True)


class vggblock(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(vggblock, self).__init__()
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, kernel_size, 2)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv(x)


class deconv(nn.Module):
    def __init__(self, num_out_layers, kernel_size, stride):
        super(deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(num_out_layers, num_out_layers, kernel_size, stride, padding=1,
                                         output_padding=1)

    def forward(self, x):
        p_x = F.pad(x, (1, 1, 1, 1))
        x = self.deconv(p_x)
        x = x[:, :, 3:-1, 3:-1]
        return x


class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 2, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        x = self.normalize(x)
        return 0.3 * self.sigmoid(x)


def scale_pyramid(self, img, num_scales):
    scaled_imgs = [img]
    _, h, w = img.size()
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh, nw = h // ratio, w // ratio
        scaled_imgs.append(F.interpolate(img, size=(nh, nw), mode='area'))
    return scaled_imgs


class VGG16_md(nn.Module):
    """encoder block"""

    def __init__(self, num_in_layers, use_deconv=False):
        super(VGG16_md, self).__init__()
        self.conv1 = vggblock(num_in_layers, 32, 7)
        self.conv2 = vggblock(32, 64, 5)
        self.conv3 = vggblock(64, 128, 3)
        self.conv4 = vggblock(128, 256, 3)
        self.conv5 = vggblock(256, 512, 3)
        self.conv6 = vggblock(256, 512, 3)
        self.conv7 = vggblock(256, 512, 3)

        if use_deconv:
            self.upconv = deconv
        else:
            self.upconv = upconv

        self.upconv7 = self.upconv(512, 512, 3, 2)
        self.iconv = conv(512 + 512, 512, 3, 1)

        self.upconv6 = self.upconv(512, 512, 3, 2)
        self.iconv6 = conv(256 + 512, 512, 3, 1)

        self.upconv5 = self.upconv(512, 256, 3, 2)
        self.iconv5 = conv(128 + 256, 256, 3, 1)

        self.upconv4 = self.upconv(256, 128, 3, 2)
        self.iconv4 = conv(64 + 128, 128, 3, 1)
        self.disp4 = get_disp(128)

        self.upconv3 = self.upconv(128, 64, 3, 2)
        self.iconv3 = conv(64 + 64 + 2, 64, 3, 1)
        self.disp3 = get_disp(64)

        self.upconv2 = self.upconv(64, 32, 3, 2)
        self.iconv2 = conv(64 + 32 + 2, 32, 3, 1)
        self.disp2 = get_disp(32)

        self.upconv1 = self.upconv(32, 16, 3, 2)
        self.iconv1 = conv(16 + 2, 16, 3, 1)
        self.disp1 = get_disp(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)

        skip1 = x1
        skip2 = x2
        skip3 = x3
        skip4 = x4
        skip5 = x5
        skip6 = x6

        upconv7 = self.upconv7(x7)  # H/64
        concat7 = torch.cat((upconv7, skip6), 1)
        iconv7 = self.iconv(concat7)

        upconv6 = self.upconv6(iconv7)  # H/32
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = conv(concat6, 512, 3, 1)

        upconv5 = self.upconv5(iconv6)  # H/16
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = conv(concat5, 256, 3, 1)

        upconv4 = self.upconv4(iconv5)  # H/8
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = conv(concat4, 128, 3, 1)
        self.disp4 = self.disp4(iconv4)
        self.udisp4 = nn.functional.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)

        upconv3 = self.upconv33(iconv4)  # H/4
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = conv(concat3, 64, 3, 1)
        self.disp3 = self.disp3(iconv3)
        self.udisp3 = nn.functional.interpolate(self.disp3, scale_factor=2, mode='bilinear', align_corners=True)

        upconv2 = self.upconv2(iconv3)  # H/2
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = conv(concat2, 32, 3, 1)
        self.disp2 = self.disp2(iconv2)
        self.udisp2 = nn.functional.interpolate(self.disp2, scale_factor=2, mode='bilinear', align_corners=True)

        upconv1 = self.upconv1(iconv2)  # H
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = conv(concat1, 16, 3, 1)
        self.disp1 = self.disp1(iconv1)

        return self.disp1, self.disp2, self.disp3, self.disp4
