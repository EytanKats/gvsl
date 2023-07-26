from monai.losses import DiceLoss


def get_loss_functions(settings):

    loss_fn = DiceLoss(
        include_background=settings['loss']['include_background'],
        to_onehot_y=settings['loss']['to_onehot_y'],
        sigmoid=settings['loss']['sigmoid']
    )

    loss_fn.__name__ = settings['loss']['loss_name']

    return [loss_fn]
