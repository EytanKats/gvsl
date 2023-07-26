import torch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction


class DiceMetricWrapper:

    def __init__(self, settings, mode):
        self.settings = settings
        self.mode = mode

        self.label_processor = AsDiscrete(to_onehot=settings['model']['out_channels'])
        self.predictions_processor = AsDiscrete(argmax=True, to_onehot=settings['model']['out_channels'])
        self.metric = DiceMetric(
            include_background=settings['metrics']['include_background'],
            reduction=MetricReduction.MEAN_BATCH
        )

    def __call__(self, y_pred, y_true):

        labels = [self.label_processor(label) for label in y_true]
        predictions = [self.predictions_processor(prediction) for prediction in y_pred]

        self.metric.reset()
        self.metric(y_pred=predictions, y=labels)
        metric = self.metric.aggregate()

        if self.mode == 'mean':
            res = torch.mean(metric)
        else:
            metric_idx = self.settings['dataset']['labels'][self.mode]
            res = metric[metric_idx]

        return res
