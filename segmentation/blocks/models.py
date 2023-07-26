import torch
from Downstream.models_finetune.UNet_GVSL import UNet3D_GVSL as UNet3D_GVSL_finetune
from Downstream.models_linear.UNet_GVSL import UNet3D_GVSL as UNet3D_GVSL_linear


def get_model(settings):

    if settings['model']['architecture'] == 'unet3d_gvsl_finetune':
        model = UNet3D_GVSL_finetune(
            n_classes=settings['model']['out_channels'],
            pretrained_weights=settings['model']['pretrained_weights']
        )
    elif settings['model']['architecture'] == 'unet3d_gvsl_linear':
        model = UNet3D_GVSL_linear(
            n_classes=settings['model']['out_channels'],
            pretrained_weights=settings['model']['pretrained_weights']
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model
