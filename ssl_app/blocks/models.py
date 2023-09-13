import torch

from ssl_app.core.unet3d_gvsl import DoubleConv
from ssl_app.core.unet3d_gvsl import UNet3D_GVSL


class ElasticRegistrationHead(torch.nn.Module):

    def __init__(self):

        super(ElasticRegistrationHead, self).__init__()

        self.registration_conv = DoubleConv(64, 16)
        self.registration_out_conv = torch.nn.Conv3d(16, 3, 3, padding=1)

    def forward(self, target_features, source_features):
        x = torch.cat([target_features, source_features], dim=1)
        x = self.registration_conv(x)
        x = self.registration_out_conv(x)

        return x


class RestorationHead(torch.nn.Module):

    def __init__(self):
        super(RestorationHead, self).__init__()

        self.restoration_conv = torch.nn.Sequential(
            torch.nn.Conv3d(32, 16, 3, padding=1),
            torch.nn.GroupNorm(16 // 4, 16),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(16, 1, 1))

    def forward(self, features):
        x = self.restoration_conv(features)
        return x


def get_feature_extractor(settings):

    model = UNet3D_GVSL(
        n_channels=1,
        chs=(32, 64, 128, 256, 512, 256, 128, 64, 32)
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def get_elastic_registration_head(settings):

    model = ElasticRegistrationHead()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def get_restoration_head(settings):

    model = RestorationHead()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def get_model(settings):

    architectures = {
        'feature_extractor': get_feature_extractor,
        'elastic_registration_head': get_elastic_registration_head,
        'restoration_head': get_restoration_head,
    }

    return architectures
