import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../simple_converge/')

import os
# Set enumeration order of GPUs to be same as for 'nvidia-smi' command and choose visible GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import logging
# logging.disable(logging.WARNING)

import torch
import GPUtil as gpu

from simple_converge.manager.Manager import fit
from simple_converge.mlops.MLOpsTask import MLOpsTask

from settings import settings
from segmentation.blocks.models import get_model
from segmentation.blocks.loss_functions import get_loss_functions
from segmentation.blocks.metrics import get_metrics
from segmentation.blocks.optimizers import get_optimizer
from segmentation.blocks.data_loaders import get_data_loaders
from segmentation.blocks.applications import get_app

if __name__ == "__main__":

    torch.multiprocessing.set_sharing_strategy('file_system')

    # gpu.getFirstAvailable(order='first', attempts=200, interval=600, verbose=True,
    #                       excludeID=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

    # Create MLOps task
    mlops_task = MLOpsTask(
        settings=settings['mlops']
    )

    # Get data loaders for training, validation
    training_data_loaders, validation_data_loaders = get_data_loaders(settings)

    # Run training
    fit(
        settings,
        mlops_task=mlops_task,
        architecture=get_model,
        loss_function=get_loss_functions,
        metric=get_metrics,
        scheduler=None,
        optimizer=get_optimizer,
        app=get_app(),
        train_dataset=None,
        train_loader=training_data_loaders,
        val_dataset=None,
        val_loader=validation_data_loaders,
        test_dataset=None,
        test_loader=None,
        postprocessor=None
    )

    exit()

