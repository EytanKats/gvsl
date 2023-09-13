from torch.optim import Adam


def get_optimizer(
        settings,
        feature_extractor,
        elastic_registration_head,
        restoration_head
    ):

    params = \
        list(feature_extractor.parameters()) + \
        list(elastic_registration_head.parameters()) + \
        list(restoration_head.parameters())

    optimizer = Adam(
        params,
        lr=settings['optimizer']['lr']
    )

    return optimizer
