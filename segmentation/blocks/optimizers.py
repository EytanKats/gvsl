from torch.optim import Adam


def get_optimizer(
        settings,
        model):

    params = model.parameters()
    optimizer = Adam(
        params,
        lr=settings['optimizer']['lr']
    )

    return optimizer
