from segmentation.core.DiceMetricWrapper import DiceMetricWrapper


def get_metrics(settings):

    metrics = []

    metric = DiceMetricWrapper(settings, mode='mean')
    metric.__name__ = f'{settings["metrics"]["metric_name"]}_mean'
    metrics.append(metric)

    for label in settings['dataset']['labels']:
        metric = DiceMetricWrapper(settings, mode=label)
        metric.__name__ = f'{settings["metrics"]["metric_name"]}_{label}'
        metrics.append(metric)

    return metrics
