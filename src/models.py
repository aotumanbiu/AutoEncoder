import torch
import torch.nn as nn
from torchsummary import summary
import netron
from torchvision.models import vgg16

out_channels = [512, 512, 512, 256, 128, 64, 3]
blocks = [3, 3, 3, 2, 2, 1]


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, init_weights=True):
        super(AutoEncoder, self).__init__()

        assert in_channels == 3

        VGG16 = vgg16()
        self.encoder = nn.Sequential(*list(VGG16.children())[:-2])
        self.decoder_1 = self._make_up_layers(blocks[0], in_channels=out_channels[0], out_channels=out_channels[1])
        self.decoder_2 = self._make_up_layers(blocks[1], in_channels=out_channels[1], out_channels=out_channels[2])
        self.decoder_3 = self._make_up_layers(blocks[2], in_channels=out_channels[2], out_channels=out_channels[3])
        self.decoder_4 = self._make_up_layers(blocks[3], in_channels=out_channels[3], out_channels=out_channels[4])
        self.decoder_5 = self._make_up_layers(blocks[4], in_channels=out_channels[4], out_channels=out_channels[5])
        self.decoder_6 = self._make_up_layers(blocks[5], in_channels=out_channels[5], out_channels=out_channels[6])

        if init_weights:
            self._initialize_weights()

    def forward(self, inputs):
        x = self.encoder(inputs)  # [B, 512, 7, 7]

        x = self.decoder_1(x)  # [B, 512, 14, 14]
        x = self.decoder_2(x)  # [B, 512, 28, 28]
        x = self.decoder_3(x)  # [B, 256, 56, 56]
        x = self.decoder_4(x)  # [B, 128, 112, 112]
        x = self.decoder_5(x)  # [B, 64, 224, 224]
        x = self.decoder_6(x)  # [B, 3, 224, 224]

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
    # model = SegNet().to(device)
    # vgg = vgg16().cuda()
    # # del vgg.classifier
    # de = nn.Sequential(*list(vgg.children())[:-2])

    a = AutoEncoder().cuda()
    inputs = torch.randn((3, 3, 224, 224)).cuda()
    # path = '0.onnx'
    # torch.onnx.export(a, inputs, path)
    # netron.start(path)
    summary(model=a, input_size=(3, 224, 224))
    # new = nn.ConvTranspose2d(in_channels=3, out_channels=6, kernel_size=4, stride=2, padding=1).cuda()
    # b = a(inputs)
    # print(b)
    # print(model)

    # outs = model(inputs)
    # print(outs.shape)
    # path = '0.onnx'
    # torch.onnx.export(model, inputs, path, verbose=9)
    # netron.start(path)
