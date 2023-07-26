from functools import partial
from monai.inferers import sliding_window_inference

from simple_converge.apps import SingleModelApp


class SingleModelAppEx(SingleModelApp):

    def __init__(
            self,
            settings,
            mlops_task,
            architecture,
            loss_function,
            metric,
            scheduler,
            optimizer
    ):

        super(SingleModelAppEx, self).__init__(
            settings,
            mlops_task,
            architecture,
            loss_function,
            metric,
            scheduler,
            optimizer
        )

        self.val_model = partial(
            sliding_window_inference,
            roi_size=self.settings['model']['val_roi_size'],
            sw_batch_size=settings['model']['val_sw_batch_size'],
            predictor=self.model,
            overlap=settings['model']['overlap'],
        )

    def parse_batch_data(self, data):
        input_data = data['image']
        labels = data['label']
        return input_data, labels
