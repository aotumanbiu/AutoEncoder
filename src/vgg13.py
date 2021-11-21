import torch
import torch.nn as nn
from torchvision.models import vgg13


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        VGG13 = vgg13()
        self.encoder = nn.Sequential(*list(VGG13.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        out_channels = [512, 512, 512, 256, 128, 64, 3]
        blocks = [2, 2, 2, 2, 2, 1]

        super(Decoder, self).__init__()
        self.stage1 = self._make_up_layers(blocks[0], in_channels=out_channels[0], out_channels=out_channels[1])
        self.stage2 = self._make_up_layers(blocks[1], in_channels=out_channels[1], out_channels=out_channels[2])
        self.stage3 = self._make_up_layers(blocks[2], in_channels=out_channels[2], out_channels=out_channels[3])
        self.stage4 = self._make_up_layers(blocks[3], in_channels=out_channels[3], out_channels=out_channels[4])
        self.stage5 = self._make_up_layers(blocks[4], in_channels=out_channels[4], out_channels=out_channels[5])
        self.stage6 = self._make_up_layers(blocks[5], in_channels=out_channels[5], out_channels=out_channels[6])

    def forward(self, x):
        x = self.stage1(x)  # [B, 512, 14, 14]
        x = self.stage2(x)  # [B, 512, 28, 28]
        x = self.stage3(x)  # [B, 256, 56, 56]
        x = self.stage4(x)  # [B, 128, 112, 112]
        x = self.stage5(x)  # [B, 64, 224, 224]
        x = self.stage6(x)  # [B, 3, 224, 224]

        return x

    def _make_up_layers(self, nums, in_channels, out_channels):
        if nums == 1:
            return nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=3,
                                           padding=1),
                                 nn.Sigmoid())

        blocks = []
        upsample = nn.ConvTranspose2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=4,
                                      stride=2,
                                      padding=1)
        blocks.append(upsample)

        for i in range(nums):
            blocks.append(nn.Conv2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1))

            blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(nn.ReLU(inplace=True))

        return nn.Sequential(*list(blocks))


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, init_weights=True):
        super(AutoEncoder, self).__init__()

        assert in_channels == 3

        self.encoder = Encoder()
        self.decoder = Decoder()

        if init_weights:
            self._initialize_weights()

    def forward(self, inputs):
        x = self.encoder(inputs)  # [B, 512, 7, 7]
        x = self.decoder(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    ae = AutoEncoder(in_channels=3, init_weights=False).cuda()
    a = torch.rand((3, 3, 224, 224)).cuda()
    out = ae(a)
    print(0)
