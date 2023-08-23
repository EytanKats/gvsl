import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from exploration.gvsl import GVSL
from utils.utils import dice
from utils.STN import SpatialTransformer, AffineTransformer
from utils.Transform_self import SpatialTransform, AppearanceTransform
from Downstream.utils.monai_data_loader import get_data_loaders

class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()

        self.pretrained_weights = [
            '',
            '/mnt/share/experiments/label/gvsl/nako_1000_pretraining/GVSL_epoch_1000.pth',
            '/mnt/share/experiments/label/gvsl/original/GVSL_epoch_1000.pth'
            ]

        # Data augmentation
        self.spatial_aug = SpatialTransform(do_rotation=True,
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
                                            sigma=(10., 13.))

        # transformation for restoration
        self.style_aug = AppearanceTransform(local_rate=0.8,
                                             nonlinear_rate=0.9,
                                             paint_rate=0.9,
                                             inpaint_rate=0.2)

        # initialize model
        self.gvsl = GVSL()

        if torch.cuda.is_available():
            self.gvsl = self.gvsl.cuda()

        self.stn = SpatialTransformer()
        self.atn = AffineTransformer()

        self.dataloader_train, _, _ = get_data_loaders()

    def tsne(self):

        self.gvsl.eval()
        for data in self.dataloader_train:

            img1 = torch.unsqueeze(data['image'][0], dim=0)
            label1 = data['label'][0]

            img2 = torch.unsqueeze(data['image'][1], dim=0)
            label2 = data['label'][1]

            if torch.cuda.is_available():
                img1 = img1.cuda()
                img2 = img2.cuda()

            label1_np = label1.numpy()
            spleen_loc = np.random.choice(np.sum(label1_np[1, ...] == 1), 1000)
            kidney_loc = np.random.choice(np.sum(label1_np[2, ...] == 1), 1000)
            liver_loc = np.random.choice(np.sum(label1_np[6, ...] == 1), 1000)
            stomach_loc = np.random.choice(np.sum(label1_np[7, ...] == 1), 1000)

            for weights in self.pretrained_weights:
                if weights:
                    state_dict = torch.load(weights)
                    self.gvsl.load_state_dict(state_dict)
                res_A, warp_BA_atn, warp_BA_stn, aff_mat_BA, flow_BA, fA_l, fB_L = self.gvsl(img1, img2)

                self.show_tsne(fA_l, label1, spleen_loc, kidney_loc, liver_loc, stomach_loc)

    def show_tsne(self, features, label, spleen_loc, kidney_loc, liver_loc, stomach_loc):

        label1_np = label.numpy()
        f1_np = features.detach().cpu().numpy()[0]
        spleen_f = np.transpose(f1_np[:, label1_np[1, ...] == 1])
        kidney_f = np.transpose(f1_np[:, label1_np[2, ...] == 1])
        liver_f = np.transpose(f1_np[:, label1_np[6, ...] == 1])
        stomach_f = np.transpose(f1_np[:, label1_np[7, ...] == 1])

        spleen_f = spleen_f[spleen_loc, :]
        kidney_f = kidney_f[kidney_loc, :]
        liver_f = liver_f[liver_loc, :]
        stomach_f = stomach_f[stomach_loc, :]

        features = np.concatenate([spleen_f, kidney_f, liver_f, stomach_f])
        labels = np.concatenate([np.ones(1000) * 1, np.ones(1000) * 2, np.ones(1000) * 3, np.ones(1000) * 4])

        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(features)

        df = pd.DataFrame()
        df["y"] = labels
        df["comp-1"] = z[:, 0]
        df["comp-2"] = z[:, 1]

        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", 4),
                        data=df).set(title="T-SNE projection")
        plt.show()
        plt.close()

    def explore(self):

        self.gvsl.eval()
        for data in self.dataloader_train:

            img1 = torch.unsqueeze(data['image'][0], dim=0)
            label1 = data['label'][0]

            img2 = torch.unsqueeze(data['image'][1], dim=0)
            label2 = data['label'][1]

            img1_np = img1.data.numpy()[0, 0, ...].copy()
            for image_slice in range(63, 64):
                plt.imshow(img1_np[:, :, image_slice], cmap='gray')
                plt.colorbar()
                plt.show()
                plt.close()

            # img2_np = img2.data.numpy()[0, 0, ...].copy()
            # for image_slice in range(63, 64):
            #     plt.imshow(img2_np[:, :, image_slice], cmap='gray')
            #     plt.colorbar()
            #     plt.show()
            #     plt.close()

            if torch.cuda.is_available():
                img1 = img1.cuda()
                label1 = label1.cuda()
                img2 = img2.cuda()
                label2 = label2.cuda()

            mat, code_spa = self.spatial_aug.rand_coords(img1.shape[2:])
            img1_aug = self.spatial_aug.augment_spatial(img1, mat, code_spa)
            img1_aug_np = img1_aug.data.cpu().numpy()[0, 0, ...].copy()
            for image_slice in range(63, 64):
                plt.imshow(img1_aug_np[:, :, image_slice], cmap='gray')
                plt.colorbar()
                plt.show()
                plt.close()

            res_A, warp_BA_atn, warp_BA_stn, aff_mat_BA, flow_BA, _, _ = self.gvsl(img1, img1_aug)

            flow_BA_np = flow_BA.data.cpu().numpy()[0, 0, ...].copy()
            for image_slice in range(63, 64):
                plt.imshow(flow_BA_np[:, :, image_slice])
                plt.colorbar()
                plt.show()
                plt.close()

            warp_BA_atn_np = warp_BA_atn.data.cpu().numpy()[0, 0, ...].copy()
            for image_slice in range(63, 64):
                plt.imshow(warp_BA_atn_np[:, :, image_slice], cmap='gray')
                plt.colorbar()
                plt.show()
                plt.close()

            warp_BA_stn_np = warp_BA_stn.data.cpu().numpy()[0, 0, ...].copy()
            for image_slice in range(63, 64):
                plt.imshow(warp_BA_stn_np[:, :, image_slice], cmap='gray')
                plt.colorbar()
                plt.show()
                plt.close()

            dice_before = []
            for i in range(12):
                dice_before.append(dice(label1[i+1], label2[i+1]))
            print(np.mean(dice_before))

            dice_after = []
            for i in range(12):
                warped_label = self.stn(self.atn(torch.unsqueeze(torch.unsqueeze(label2[i+1], dim=0), dim=0), aff_mat_BA, mode='nearest'), flow_BA, mode='nearest')
                dice_after.append(dice(label1[i+1], warped_label[0][0]))
            print(np.mean(dice_after))

            print('Bingo')

    def load(self):
        print('Loading weights')
        state_dict = torch.load(self.pretrained_weights)
        self.gvsl.load_state_dict(state_dict)

if __name__ == '__main__':
    trainer = Trainer()
    # trainer.load()
    trainer.tsne()



