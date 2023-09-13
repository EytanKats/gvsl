import os
import torch
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

from simple_converge.apps.BaseApp import BaseApp
from ssl_app.core.plots import show_images, show_tsne
from ssl_app.core.elastic_transform import ElasticTransform
from ssl_app.core.spatial_augmentations import SpatialTransform
from ssl_app.core.intensity_augmentations import AppearanceTransform


class DenseSslApp(BaseApp):

    """
    This class defines self-supervised application for dense representation learning.
    The application contains single feature extractor model and few task specific heads.
    """

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
        """
        This method initializes parameters
        :return: None
        """

        super(DenseSslApp, self).__init__(
            settings,
            mlops_task,
            loss_function,
            metric,
        )

        self.style_aug = AppearanceTransform(
            local_rate=0.8,
            nonlinear_rate=0.9,
            paint_rate=0.9,
            inpaint_rate=0.2
        )

        self.spatial_aug = SpatialTransform(
            do_rotation=True,
            angle_x=(-np.pi / 9, np.pi / 9),
            angle_y=(-np.pi / 9, np.pi / 9),
            angle_z=(-np.pi / 9, np.pi / 9),
            do_scale=True,
            scale_x=(0.75, 1.25),
            scale_y=(0.75, 1.25),
            scale_z=(0.75, 1.25),
            do_translate=True,
            trans_x=(-0.1, 0.1),
            trans_y=(-0.1, 0.1),
            trans_z=(-0.1, 0.1),
            do_shear=True,
            shear_xy=(-np.pi / 18, np.pi / 18),
            shear_xz=(-np.pi / 18, np.pi / 18),
            shear_yx=(-np.pi / 18, np.pi / 18),
            shear_yz=(-np.pi / 18, np.pi / 18),
            shear_zx=(-np.pi / 18, np.pi / 18),
            shear_zy=(-np.pi / 18, np.pi / 18),
            do_elastic_deform=True,
            alpha=(0., 512.),
            sigma=(10., 13.)
        )

        self.elastic_transform = ElasticTransform()

        # Create models' architecture
        if architecture is not None:
            architectures = architecture(settings)
            self.feature_extractor = architectures['feature_extractor'](settings)
            self.elastic_registration_head = architectures['elastic_registration_head'](settings)
            self.restoration_head = architectures['restoration_head'](settings)
        else:
            self.feature_extractor = None
            self.elastic_registration_head = None
            self.restoration_head = None

        if optimizer is not None:
            # Instantiate optimizer for base encoder and predictor but not for the momentum encoder
            self.optimizer = optimizer(
                settings,
                self.feature_extractor,
                self.elastic_registration_head,
                self.restoration_head
            )
        else:
            self.optimizer = None

        if scheduler is not None:
            self.scheduler = scheduler(settings, self.optimizer)
        else:
            self.scheduler = None

        self.ckpt_cnt = 0
        self.latest_ckpt_path = None

        self.vis_dir = os.path.join(settings['manager']['output_folder'], '0', 'vis')
        os.makedirs(self.vis_dir)

    def restore_ckpt(self, ckpt_path=''):

        if ckpt_path:
            path_to_restore = ckpt_path
        else:
            path_to_restore = self.latest_ckpt_path

        logger.info(f'Restore checkpoint {path_to_restore}')
        checkpoint = torch.load(ckpt_path)

        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.elastic_registration_head.load_state_dict(checkpoint['elastic_registration_head'])
        self.restoration_head.load_state_dict(checkpoint['restoration_head'])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_ckpt(self, ckpt_path):

        self.latest_ckpt_path = ckpt_path + '-' + str(self.ckpt_cnt) + '.pth'
        self.ckpt_cnt += 1

        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'elastic_registration_head': self.elastic_registration_head.state_dict(),
            'restoration_head': self.restoration_head.state_dict(),
            'optimizer': self.optimizer.state_dict()
        },
            self.latest_ckpt_path
        )

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return {'base_lr': param_group['lr']}

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def on_epoch_start(self):
        pass

    def on_epoch_end(self, is_plateau=False):
        pass

    def training_step(self, data, epoch, cur_iteration, iterations_per_epoch):

        # Set models to training mode
        self.feature_extractor.train()
        self.elastic_registration_head.train()
        self.restoration_head.train()

        # Get image
        image_original = data

        # Apply intensity augmentations to get first image view
        image_intensity_augmented = self.style_aug.rand_aug(image_original[0, 0, ...].numpy().copy())
        image_intensity_augmented[image_original[0, 0, ...].numpy() < 1e-2] = 0
        image_intensity_augmented = torch.from_numpy(image_intensity_augmented[np.newaxis, np.newaxis, :, :, :])

        # Send images to device
        image_original = image_original.to(self.device)
        image_intensity_augmented = image_intensity_augmented.to(self.device)

        # Apply spatial augmentations to get second image view
        mat, code_spa = self.spatial_aug.rand_coords(image_original.shape[2:])
        image_spatially_augmented = self.spatial_aug.augment_spatial(image_original, code_spa=code_spa)

        # image_original_np = image_original.data.cpu().numpy()[0, 0, ...].copy()
        # center_slice = image_original_np.shape[2] // 2
        # for image_slice in range(center_slice, center_slice + 1):
        #     plt.imshow(image_original_np[:, :, image_slice], cmap='gray')
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()
        #
        # image_intensity_augmented_np = image_intensity_augmented.data.cpu().numpy()[0, 0, ...].copy()
        # center_slice = image_intensity_augmented_np.shape[2] // 2
        # for image_slice in range(center_slice, center_slice + 1):
        #     plt.imshow(image_intensity_augmented_np[:, :, image_slice], cmap='gray')
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()
        #
        # image_spatially_augmented_np = image_spatially_augmented.data.cpu().numpy()[0, 0, ...].copy()
        # center_slice = image_spatially_augmented_np.shape[2] // 2
        # for image_slice in range(center_slice, center_slice + 1):
        #     plt.imshow(image_spatially_augmented_np[:, :, image_slice], cmap='gray')
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()

        # Extract features
        _, features_original = self.feature_extractor(image_original)
        _, features_intensity_augmented = self.feature_extractor(image_intensity_augmented)
        _, features_spatially_augmented = self.feature_extractor(image_spatially_augmented)

        # Warp spatially augmented version to the original image
        flow = self.elastic_registration_head(features_original, features_spatially_augmented)
        image_warped = self.elastic_transform(image_spatially_augmented, flow)

        # Restore original image from its intensity augmented version
        image_restored = self.restoration_head(features_intensity_augmented)

        # Calculate loss
        loss_restoration = self.losses_fns[0](image_restored, image_original)
        loss_registration = self.losses_fns[1](image_warped, image_original)
        loss_flow_regularization = self.losses_fns[2](flow)

        loss = \
            self.settings['loss']['weight_restoration'] * loss_restoration + \
            self.settings['loss']['weight_registration'] * loss_registration + \
            self.settings['loss']['weight_flow_regularization'] * loss_flow_regularization

        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Prepare loss and metrics to logging
        batch_loss_list = list()
        batch_loss_list.append(loss_restoration.detach().cpu().numpy())
        batch_loss_list.append(loss_registration.detach().cpu().numpy())
        batch_loss_list.append(loss_flow_regularization.detach().cpu().numpy())

        batch_metric_list = list()

        return batch_loss_list, batch_metric_list

    def validation_step(self, data, epoch, cur_iteration, iterations_per_epoch):

        # Set models to evaluation mode
        self.feature_extractor.eval()
        self.elastic_registration_head.eval()
        self.restoration_head.eval()

        # Get image and label
        image_original = data['image']
        label = data['label']

        # Apply intensity augmentations to get first image view
        image_intensity_augmented = self.style_aug.rand_aug(image_original[0, 0, ...].numpy().copy())
        image_intensity_augmented[image_original[0, 0, ...].numpy() < 1e-2] = 0
        image_intensity_augmented = torch.from_numpy(image_intensity_augmented[np.newaxis, np.newaxis, :, :, :])

        # Send images and label to device
        image_original = image_original.to(self.device)
        image_intensity_augmented = image_intensity_augmented.to(self.device)
        label = label.to(self.device)

        # Apply spatial augmentations to get second image view
        mat, code_spa = self.spatial_aug.rand_coords(image_original.shape[2:])
        image_spatially_augmented = self.spatial_aug.augment_spatial(image_original, code_spa=code_spa)

        with torch.no_grad():

            # Extract features
            _, features_original = self.feature_extractor(image_original)
            _, features_intensity_augmented = self.feature_extractor(image_intensity_augmented)
            _, features_spatially_augmented = self.feature_extractor(image_spatially_augmented)

            # Warp spatially augmented version to the original image
            flow = self.elastic_registration_head(features_original, features_spatially_augmented)
            image_warped = self.elastic_transform(image_spatially_augmented, flow)

            # Restore original image from its intensity augmented version
            image_restored = self.restoration_head(features_intensity_augmented)

            # Calculate loss
            loss_restoration = self.losses_fns[0](image_restored, image_original)
            loss_registration = self.losses_fns[1](image_warped, image_original)
            loss_flow_regularization = self.losses_fns[2](flow)

        show_images(image_original, image_intensity_augmented, 'intensity augmentation', epoch, cur_iteration,
                    self.vis_dir, self.mlops_task)
        show_images(image_original, image_spatially_augmented, 'spatial augmentation', epoch, cur_iteration,
                    self.vis_dir, self.mlops_task)
        show_images(image_intensity_augmented, image_restored, 'imgage restored', epoch, cur_iteration,
                    self.vis_dir, self.mlops_task)
        show_images(image_original, image_warped, 'image warped', epoch, cur_iteration,
                    self.vis_dir, self.mlops_task)
        show_tsne(features_original, label, 'tsne', epoch, cur_iteration,
                  self.vis_dir, self.mlops_task)

        # Prepare loss and metrics to logging
        batch_loss_list = list()
        batch_loss_list.append(loss_restoration.detach().cpu().numpy())
        batch_loss_list.append(loss_registration.detach().cpu().numpy())
        batch_loss_list.append(loss_flow_regularization.detach().cpu().numpy())

        batch_metric_list = list()

        return batch_loss_list, batch_metric_list

    def predict(self, data):
        pass

    def summary(self):
        logger.info("Feature extractor architecture:")
        print(self.feature_extractor)

        logger.info("Elastic registration head architecture:")
        print(self.elastic_registration_head)

        logger.info("Reconstraction head architecture:")
        print(self.restoration_head)
