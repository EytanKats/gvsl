import os
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as functional

from simple_converge.apps.BaseApp import BaseApp
from ssl_app.core.plots import show_images, show_tsne
from ssl_app.core.coupled_convex import coupled_convex
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
        image_1 = data[0].unsqueeze(0)
        image_2 = data[1].unsqueeze(0)

        # Apply intensity augmentations to get first image view
        image_1_intensity_augmented = self.style_aug.rand_aug(image_1[0, 0, ...].numpy().copy())
        image_1_intensity_augmented[image_1[0, 0, ...].numpy() < 1e-2] = 0
        image_1_intensity_augmented = torch.from_numpy(image_1_intensity_augmented[np.newaxis, np.newaxis, :, :, :])

        # Send images to device
        image_1 = image_1.to(self.device)
        image_1_intensity_augmented = image_1_intensity_augmented.to(self.device)
        image_2 = image_2.to(self.device)

        # Apply spatial augmentations to get second image view
        # mat, code_spa = self.spatial_aug.rand_coords(image_1.shape[2:])
        # image_1_spatially_augmented = self.spatial_aug.augment_spatial(image_1, code_spa=code_spa)

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
        shallow_features_1, local_features_1 = self.feature_extractor(image_1)
        shallow_features_2, local_features_2 = self.feature_extractor(image_2)
        _, local_features_1_intensity_augmented = self.feature_extractor(image_1_intensity_augmented)
        # _, local_features_1_spatially_augmented = self.feature_extractor(image_1_spatially_augmented)

        # Warp 1st and 2nd images
        global_registration_flow = coupled_convex(feat_fix=shallow_features_1, feat_mov=shallow_features_2, use_ice=False, img_shape=image_1.shape[2:])
        # grid = functional.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(), (1, 1, image_1.shape[2], image_1.shape[3], image_1.shape[4]))
        # image_2_warped = functional.grid_sample(image_2, grid + global_registration_flow.permute(0, 2, 3, 4, 1), mode='bilinear')
        grid = functional.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(), (1, 1, local_features_1.shape[2], local_features_1.shape[3], local_features_1.shape[4]))
        local_features_2_warped = functional.grid_sample(local_features_2, grid + global_registration_flow.permute(0, 2, 3, 4, 1), mode='bilinear')

        # Warp spatially augmented version to the original image
        # local_registration_flow = self.elastic_registration_head(local_features_1, local_features_1_spatially_augmented)
        # image_1_spatially_augmented_warped = self.elastic_transform(image_1_spatially_augmented, local_registration_flow)
        local_registration_flow = self.elastic_registration_head(local_features_1, local_features_2_warped)

        grid = functional.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(),(1, 1, image_2.shape[2], image_2.shape[3], image_2.shape[4]))
        image_2_globally_warped = functional.grid_sample(image_2, grid + global_registration_flow.permute(0, 2, 3, 4, 1), mode='bilinear')
        image_2_locally_warped = self.elastic_transform(image_2_globally_warped, local_registration_flow)

        # Restore original image from its intensity augmented version
        image_1_intensity_augmented_restored = self.restoration_head(local_features_1_intensity_augmented)

        # Calculate loss
        loss_restoration = self.losses_fns[0](image_1_intensity_augmented_restored, image_1)
        loss_global_registration = self.losses_fns[1](image_2_locally_warped, image_1)
        # loss_local_registration = self.losses_fns[2](image_1_spatially_augmented_warped, image_1)
        loss_flow_regularization = self.losses_fns[3](local_registration_flow)

        loss = \
            self.settings['loss']['weight_restoration'] * loss_restoration + \
            self.settings['loss']['weight_global_registration'] * loss_global_registration + \
            self.settings['loss']['weight_flow_regularization'] * loss_flow_regularization

        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Prepare loss and metrics to logging
        batch_loss_list = list()
        batch_loss_list.append(loss_restoration.detach().cpu().numpy())
        batch_loss_list.append(loss_global_registration.detach().cpu().numpy())
        batch_loss_list.append(0)
        batch_loss_list.append(loss_flow_regularization.detach().cpu().numpy())

        batch_metric_list = list()

        return batch_loss_list, batch_metric_list

    def validation_step(self, data, epoch, cur_iteration, iterations_per_epoch):

        # Set models to evaluation mode
        self.feature_extractor.eval()
        self.elastic_registration_head.eval()
        self.restoration_head.eval()

        # Get image and label
        image_1 = data['image'][0].unsqueeze(0)
        image_2 = data['image'][1].unsqueeze(0)
        label = data['label'][0].unsqueeze(0)

        # Apply intensity augmentations to get first image view
        image_1_intensity_augmented = self.style_aug.rand_aug(image_1[0, 0, ...].numpy().copy())
        image_1_intensity_augmented[image_1[0, 0, ...].numpy() < 1e-2] = 0
        image_1_intensity_augmented = torch.from_numpy(image_1_intensity_augmented[np.newaxis, np.newaxis, :, :, :])

        # Send images to device
        image_1 = image_1.to(self.device)
        image_1_intensity_augmented = image_1_intensity_augmented.to(self.device)
        image_2 = image_2.to(self.device)

        # Send images and label to device
        image_1 = image_1.to(self.device)
        image_1_intensity_augmented = image_1_intensity_augmented.to(self.device)
        image_2 = image_2.to(self.device)
        label = label.to(self.device)

        # Apply spatial augmentations to get second image view
        # mat, code_spa = self.spatial_aug.rand_coords(image_1.shape[2:])
        # image_1_spatially_augmented = self.spatial_aug.augment_spatial(image_1, code_spa=code_spa)

        with torch.no_grad():
            # Extract features
            shallow_features_1, local_features_1 = self.feature_extractor(image_1)
            shallow_features_2, local_features_2 = self.feature_extractor(image_2)
            _, local_features_1_intensity_augmented = self.feature_extractor(image_1_intensity_augmented)
            # _, local_features_1_spatially_augmented = self.feature_extractor(image_1_spatially_augmented)

            # Warp 1st and 2nd images
            global_registration_flow = coupled_convex(feat_fix=shallow_features_1, feat_mov=shallow_features_2, use_ice=False, img_shape=image_1.shape[2:])
            # grid = functional.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(), (1, 1, image_1.shape[2], image_1.shape[3], image_1.shape[4]))
            # image_2_warped = functional.grid_sample(image_2, grid + global_registration_flow.permute(0, 2, 3, 4, 1), mode='bilinear')
            grid = functional.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(), ( 1, 1, local_features_1.shape[2], local_features_1.shape[3], local_features_1.shape[4]))
            local_features_2_warped = functional.grid_sample(local_features_2, grid + global_registration_flow.permute(0, 2, 3, 4, 1), mode='bilinear')

            # Warp spatially augmented version to the original image
            # local_registration_flow = self.elastic_registration_head(local_features_1, local_features_1_spatially_augmented)
            # image_1_spatially_augmented_warped = self.elastic_transform(image_1_spatially_augmented, local_registration_flow)
            local_registration_flow = self.elastic_registration_head(local_features_1, local_features_2_warped)

            grid = functional.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(), (1, 1, image_2.shape[2], image_2.shape[3], image_2.shape[4]))
            image_2_globally_warped = functional.grid_sample(image_2, grid + global_registration_flow.permute(0, 2, 3, 4, 1), mode='bilinear')
            image_2_locally_warped = self.elastic_transform(image_2_globally_warped, local_registration_flow)

            # Restore original image from its intensity augmented version
            image_1_intensity_augmented_restored = self.restoration_head(local_features_1_intensity_augmented)

            # Calculate loss
            loss_restoration = self.losses_fns[0](image_1_intensity_augmented_restored, image_1)
            loss_global_registration = self.losses_fns[1](image_2_locally_warped, image_1)
            # loss_local_registration = self.losses_fns[2](image_1_spatially_augmented_warped, image_1)
            loss_flow_regularization = self.losses_fns[3](local_registration_flow)

        show_images(image_1, image_1_intensity_augmented, 'intensity augmentation', epoch, cur_iteration, self.vis_dir, self.mlops_task)
        # show_images(image_1, image_1_spatially_augmented, 'spatial augmentation', epoch, cur_iteration, self.vis_dir, self.mlops_task)
        show_images(image_1_intensity_augmented, image_1_intensity_augmented_restored, 'restoration', epoch, cur_iteration, self.vis_dir, self.mlops_task)
        show_images(image_1, image_2_globally_warped, 'global registration', epoch, cur_iteration, self.vis_dir, self.mlops_task)
        show_images(image_1, image_2_locally_warped, 'local registration', epoch, cur_iteration, self.vis_dir, self.mlops_task)
        show_tsne(local_features_1, label, 'tsne', epoch, cur_iteration, self.vis_dir, self.mlops_task)

        # Prepare loss and metrics to logging
        batch_loss_list = list()
        batch_loss_list.append(loss_restoration.detach().cpu().numpy())
        batch_loss_list.append(loss_global_registration.detach().cpu().numpy())
        batch_loss_list.append(0)
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
