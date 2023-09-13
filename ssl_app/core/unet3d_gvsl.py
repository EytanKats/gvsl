import torch


class DoubleConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.GroupNorm(out_channels//8, out_channels),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.GroupNorm(out_channels//8, out_channels),
            torch.nn.LeakyReLU(0.2)
        )

    def forward(self, x):

        return self.double_conv(x)


class Down(torch.nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):

        return self.maxpool_conv(x)


class Up(torch.nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class UNet3D_GVSL(torch.nn.Module):

    def __init__(
            self,
            n_channels,
            chs=(32, 64, 128, 256, 512, 256, 128, 64, 32)
    ):
        super(UNet3D_GVSL, self).__init__()

        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, chs[0])

        self.down1 = Down(chs[0], chs[1])
        self.down2 = Down(chs[1], chs[2])
        self.down3 = Down(chs[2], chs[3])
        self.down4 = Down(chs[3], chs[4])

        self.up1 = Up(chs[4] + chs[3], chs[5])
        self.up2 = Up(chs[5] + chs[2], chs[6])
        self.up3 = Up(chs[6] + chs[1], chs[7])
        self.up4 = Up(chs[7] + chs[0], chs[8])

        self.__init_weight()

    def __init_weight(self):

        for m in self.modules():

            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, torch.nn.GroupNorm):
                m.weight.data.fill_(1)

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x5, x