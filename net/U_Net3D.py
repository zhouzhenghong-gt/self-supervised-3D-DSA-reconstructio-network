import torch.nn as nn
import torch.nn.functional as F
from .unet_utils import UnetConv3, UnetUp3
from .unet_networks_other import init_weights

class unet_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=2, is_batchnorm=True):
        super(unet_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp3(filters[4], filters[3], self.is_deconv, is_batchnorm)
        self.up_concat3 = UnetUp3(filters[3], filters[2], self.is_deconv, is_batchnorm)
        self.up_concat2 = UnetUp3(filters[2], filters[1], self.is_deconv, is_batchnorm)
        self.up_concat1 = UnetUp3(filters[1], filters[0], self.is_deconv, is_batchnorm)

        self.final = nn.Sequential(
            nn.Conv3d(filters[0], n_classes, 1),
        )

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        del inputs
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        del maxpool1
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        del maxpool2
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        del maxpool3
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        del maxpool4
        up4 = self.up_concat4(conv4, center)
        del center
        up3 = self.up_concat3(conv3, up4)
        del up4
        up2 = self.up_concat2(conv2, up3)
        del up3
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        final = final - final.min()
        final = final / final.max()

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
