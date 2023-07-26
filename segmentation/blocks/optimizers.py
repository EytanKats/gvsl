from torch.optim import AdamW


def get_optimizer(
        settings,
        model):

    params = model.parameters()
    optimizer = AdamW(
        params,
        lr=settings['optimizer']['lr'],
        weight_decay=settings['optimizer']['weight_decay']
    )

    return optimizer
