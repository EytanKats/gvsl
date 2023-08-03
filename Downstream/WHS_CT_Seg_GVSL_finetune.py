import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../simple_converge/')

import os
# Set enumeration order of GPUs to be same as for 'nvidia-smi' command and choose visible GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from os.path import join

import torch
from torch.utils.data import DataLoader

from models_finetune.UNet_GVSL import UNet3D_GVSL

# from utils.dataloader_heart_whs_test import DatasetFromFolder3D as DatasetFromFolder3D_test
# from utils.dataloader_nako200_whs import DatasetFromFolder3D
from utils.monai_data_loader import get_data_loaders
from utils.losses import dice_loss
from utils.utils import AverageMeter, dice, to_categorical
import numpy as np
import SimpleITK as sitk
from simple_converge.mlops.MLOpsTask import MLOpsTask

class Trainer(object):
    def __init__(self, k=0, n_classes=9, lr=1e-4, epoches=80, iters=24, batch_size=1,
                 model_name='gvsl_sslamosct240ckpt1000_mri_amos24_f2',
                 labeled_dir='',
                 test_dir='',
                 val_dir='',
                 results_dir='',
                 checkpoint_dir='/share/data_supergrover3/kats/experiments/label/gvsl/gvsl_sslamosct240ckpt1000_mri_amos24/2'):
        super(Trainer, self).__init__()

        mlops_settings = {
            'use_mlops': True,
            'project_name': 'Label',
            'task_name': 'gvsl_sslamosct240ckpt1000_mri_amos24_f2',
            'task_type': 'training',
            'tags': ['gvsl', 'sslamosct240ckpt1000', 'mri', 'amos24', 'f2'],
            'connect_arg_parser': False,
            'connect_frameworks': False,
            'resource_monitoring': True,
            'connect_streams': True
        }

        self.mlops_task = MLOpsTask(settings=mlops_settings)

        # initailize parameters
        self.k = k
        self.n_classes = n_classes
        self.epoches = epoches
        self.iters=iters
        self.lr = lr
        self.crop_shape = (128, 128, 128)
        self.labeled_dir = labeled_dir
        self.test_dir = test_dir
        self.val_dir = val_dir

        self.results_dir = results_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        # initialize networks
        self.Seger = UNet3D_GVSL(
            n_classes=n_classes,
            # pretrained_weights=None)
            pretrained_weights='/share/data_supergrover3/kats/experiments/label/gvsl/amosct_240_pretraining/GVSL_epoch_1000.pth')

        if torch.cuda.is_available():
            self.Seger = self.Seger.cuda()

        self.optS = torch.optim.Adam(self.Seger.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optS, T_max=self.epoches, eta_min=1e-5)

        # Data iterator
        # train_dataset = DatasetFromFolder3D('train', num_classes=n_classes, shape=self.crop_shape)
        # self.dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_dataset = DatasetFromFolder3D('val', num_classes=n_classes)
        # self.dataloader_val = DataLoader(val_dataset, batch_size=batch_size)
        # test_dataset = DatasetFromFolder3D_test(self.test_dir)
        # self.dataloader_test = DataLoader(test_dataset, batch_size=batch_size)

        self.dataloader_train, self.dataloader_val, self.dataloader_test = get_data_loaders()

        # define loss
        self.L_seg = dice_loss

        # define loss log
        self.L_seg_log = AverageMeter(name='L_seg_aug')
        self.val_dice = 0
        self.dice_all_mean = np.zeros(shape=(self.n_classes - 1,))

    def train_iterator(self, img, lab):
        seg = self.Seger(img)
        loss_seg = self.L_seg(seg, lab)
        self.L_seg_log.update(loss_seg.data, lab.size(0))
        # self.logwriter.writeLog([loss_seg.data.cpu().numpy()])

        loss = loss_seg

        loss.backward()
        self.optS.step()
        self.Seger.zero_grad()
        self.optS.zero_grad()

    def train_epoch(self, epoch):
        self.Seger.train()
        # for i in range(self.iters):
        #     img, lab, name = next(self.dataloader_train.__iter__())
        for data in self.dataloader_train:
            if torch.cuda.is_available():
                img = data['image'].cuda()
                lab = data['label'].cuda()

            self.train_iterator(img, lab)

        res = f'Epoch: {epoch + 1}/{self.epoches}, {self.L_seg_log.avg.cpu().numpy()}'
        print(res)

        self.mlops_task.log_scalar_to_mlops_server(f'Training loss', f'dice', self.L_seg_log.avg.cpu().numpy(), epoch + 1)

    def val(self, dataloader):
        self.Seger.eval()
        loss = []
        dice_all_mean = np.zeros(shape=(self.n_classes-1,))
        # for i, (img, lab, name) in enumerate(self.dataloader_val):
        #     name = name[0]
        for data in dataloader:
            img = data['image']
            lab = data['label'].numpy()[0]

            # lab = lab.data.numpy()[0]

            seg = to_categorical(self.predict(img), self.n_classes)
            dice_all = []
            for i in range(self.n_classes - 1):
                dice_all.append(dice(seg[i + 1], lab[i + 1]))
            print(np.mean(dice_all))

            dice_all_mean = dice_all_mean + np.array(dice_all)
            loss.append(np.mean(dice_all))

        dice_all_mean = dice_all_mean / len(dataloader)
        return np.mean(loss), dice_all_mean

    def test(self):
        self.Seger.eval()
        for i, (img, name) in enumerate(self.dataloader_test):
            name = name[0]
            seg = self.predict(img)
            seg = seg.astype(np.int8)
            if not os.path.exists(join(self.results_dir, self.model_name, 'seg')):
                os.makedirs(join(self.results_dir, self.model_name, 'seg'))

            s_m = sitk.GetImageFromArray(seg)
            sitk.WriteImage(s_m, join(self.results_dir, self.model_name, 'seg', name))
            print(name)

    def predict(self, image):
        image_shape = image.shape[2:]
        stride_x = self.crop_shape[0] // 2
        stride_y = self.crop_shape[1] // 2
        stride_z = self.crop_shape[2] // 2


        if image.shape[2] < self.crop_shape[0]:
            image = torch.cat([image, torch.zeros((1, 1, self.crop_shape[0] - image.shape[2], image.shape[3], image.shape[4]))],
                                   dim=2)
        if image.shape[3] < self.crop_shape[1]:
            image = torch.cat([image, torch.zeros((1, 1, image.shape[2], self.crop_shape[1] - image.shape[3], image.shape[4]))],
                                   dim=3)
        if image.shape[4] < self.crop_shape[2]:
            image = torch.cat([image, torch.zeros((1, 1, image.shape[2], image.shape[3], self.crop_shape[2] - image.shape[4]))],
                                   dim=4)

        image_shape_new = image.shape[2:]

        predict = np.zeros((1, self.n_classes, image_shape_new[0], image_shape_new[1], image_shape_new[2]))

        if torch.cuda.is_available():
            image = image.cuda()

        with torch.no_grad():
            for i in range(image_shape_new[0] // stride_x - 1):
                for j in range(image_shape_new[1] // stride_y - 1):
                    for k in range(image_shape_new[2] // stride_z - 1):
                        image_i = image[:, :, i * stride_x:i * stride_x + self.crop_shape[0],
                                  j * stride_y:j * stride_y + self.crop_shape[1],
                                  k * stride_z:k * stride_z + self.crop_shape[2]]
                        output = self.Seger(image_i).data.cpu().numpy()
                        predict[:, :, i * stride_x:i * stride_x + self.crop_shape[0], j * stride_y:j * stride_y + self.crop_shape[1],
                        k * stride_z:k * stride_z + self.crop_shape[2]] += output

                    image_i = image[:, :, i * stride_x:i * stride_x + self.crop_shape[0],
                              j * stride_y:j * stride_y + self.crop_shape[1],
                              image_shape_new[2] - self.crop_shape[2]:image_shape_new[2]]
                    output = self.Seger(image_i).data.cpu().numpy()
                    predict[:, :, i * stride_x:i * stride_x + self.crop_shape[0], j * stride_y:j * stride_y + self.crop_shape[1],
                    image_shape_new[2] - self.crop_shape[2]:image_shape_new[2]] += output

                for k in range(image_shape_new[2] // stride_z - 1):
                    image_i = image[:, :, i * stride_x:i * stride_x + self.crop_shape[0],
                              image_shape_new[1] - self.crop_shape[1]:image_shape_new[1],
                              k * stride_z:k * stride_z + self.crop_shape[2]]
                    output = self.Seger(image_i).data.cpu().numpy()
                    predict[:, :, i * stride_x:i * stride_x + self.crop_shape[0],
                    image_shape_new[1] - self.crop_shape[1]:image_shape_new[1],
                    k * stride_z:k * stride_z + self.crop_shape[2]] += output

                image_i = image[:, :, i * stride_x:i * stride_x + self.crop_shape[0],
                          image_shape_new[1] - self.crop_shape[1]:image_shape_new[1],
                          image_shape_new[2] - self.crop_shape[2]:image_shape_new[2]]
                output = self.Seger(image_i).data.cpu().numpy()
                predict[:, :, i * stride_x:i * stride_x + self.crop_shape[0],
                image_shape_new[1] - self.crop_shape[1]:image_shape_new[1],
                image_shape_new[2] - self.crop_shape[2]:image_shape_new[2]] += output

            for j in range(image_shape_new[1] // stride_y - 1):
                for k in range((image_shape_new[2] - self.crop_shape[2]) // stride_z):
                    image_i = image[:, :, image_shape_new[0] - self.crop_shape[0]:image_shape_new[0],
                              j * stride_y:j * stride_y + self.crop_shape[1],
                              k * stride_z:k * stride_z + self.crop_shape[2]]
                    output = self.Seger(image_i).data.cpu().numpy()
                    predict[:, :, image_shape_new[0] - self.crop_shape[0]:image_shape_new[0],
                    j * stride_y:j * stride_y + self.crop_shape[1],
                    k * stride_z:k * stride_z + self.crop_shape[2]] += output

                image_i = image[:, :, image_shape_new[0] - self.crop_shape[0]:image_shape_new[0],
                          j * stride_y:j * stride_y + self.crop_shape[1],
                          image_shape_new[2] - self.crop_shape[2]:image_shape_new[2]]
                output = self.Seger(image_i).data.cpu().numpy()
                predict[:, :, image_shape_new[0] - self.crop_shape[0]:image_shape_new[0],
                j * stride_y:j * stride_y + self.crop_shape[1],
                image_shape_new[2] - self.crop_shape[2]:image_shape_new[2]] += output

            for k in range(image_shape_new[2] // stride_z - 1):
                image_i = image[:, :, image_shape_new[0] - self.crop_shape[0]:image_shape_new[0],
                          image_shape_new[1] - self.crop_shape[1]:image_shape_new[1],
                          k * stride_z:k * stride_z + self.crop_shape[2]]
                output = self.Seger(image_i).data.cpu().numpy()
                predict[:, :, image_shape_new[0] - self.crop_shape[0]:image_shape_new[0],
                image_shape_new[1] - self.crop_shape[1]:image_shape_new[1],
                k * stride_z:k * stride_z + self.crop_shape[2]] += output

            image_i = image[:, :, image_shape_new[0] - self.crop_shape[0]:image_shape_new[0],
                      image_shape_new[1] - self.crop_shape[1]:image_shape_new[1],
                      image_shape_new[2] - self.crop_shape[2]:image_shape_new[2]]

            output = self.Seger(image_i).data.cpu().numpy()
            predict[:, :, image_shape_new[0] - self.crop_shape[0]:image_shape_new[0],
            image_shape_new[1] - self.crop_shape[1]:image_shape_new[1],
            image_shape_new[2] - self.crop_shape[2]:image_shape_new[2]] += output

            predict = np.argmax(predict[0], axis=0)[:image_shape[0], :image_shape[1], :image_shape[2]]

        return predict

    def checkpoint(self, epoch):
        torch.save(self.Seger.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, self.model_name, epoch+self.k))

    def load(self):
        self.Seger.load_state_dict(
            torch.load('{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, self.model_name, str(self.k))))

    def load_best(self):
        self.Seger.load_state_dict(
            torch.load('{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, self.model_name, 'val_dice_best')))

    def train(self):
        # self.logwriter = LogWriter(name=self.model_name, head=[self.L_seg_log.name])
        # self.val_logwriter = LogWriter(name="val_" + self.model_name, head=['Dice'])
        val_dice_best = 0
        for epoch in range(self.epoches - self.k):
            self.L_seg_log.reset()
            self.train_epoch(epoch + self.k)
            if epoch%1 == 0:
                val_dice, dice_all_mean = self.val(self.dataloader_val)
                # self.val_logwriter.writeLog([val_dice])
                if val_dice_best <= val_dice:
                    val_dice_best = val_dice
                    torch.save(self.Seger.state_dict(), '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, self.model_name, 'val_dice_best'))

                self.val_dice = 0.7 * self.val_dice + 0.3 * val_dice
                self.dice_all_mean = 0.7 * self.dice_all_mean + 0.3 * dice_all_mean

                print(f'ema dice: {self.val_dice}')
                self.mlops_task.log_scalar_to_mlops_server(f'Validation dice', f'mean_dice', self.val_dice, epoch + 1)

                # for idx, label in enumerate(['liver', 'spleen', 'right_kidney', 'left_kidney', 'pancreas']):
                # for idx, label in enumerate(['1_spleen', '2_kidney_right', '3_kidney_left', '4_gallbladder',
                #                              '5_esophagus', '6_liver', '7_stomach', '8_aorta', '9_postcava',
                #                              '10_pancreas', '11_adrenal_gland_right', '12_adrenal_gland_left']):
                for idx, label in enumerate(['1_spleen', '2_kidney_right', '3_kidney_left',
                                             '6_liver', '7_stomach', '8_aorta', '9_postcava',
                                            '10_pancreas']):
                    print(f'mean dice for {label}: {self.dice_all_mean[idx]}')
                    self.mlops_task.log_scalar_to_mlops_server(f'Validation dice', f'dice_{label}', self.dice_all_mean[idx], epoch + 1)

            self.checkpoint(self.epoches - self.k)
            self.scheduler.step()

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    trainer = Trainer()
    trainer.train()
    trainer.load_best()
    val_dice, dice_all_mean = trainer.val(trainer.dataloader_test)

    print(f'-------------------------')
    print(f'TEST RESULTS ')
    print(f'MEAN DICE: {val_dice}')
    for idx, label in enumerate(['1_spleen', '2_kidney_right', '3_kidney_left',
                                 '6_liver', '7_stomach', '8_aorta', '9_postcava',
                                 '10_pancreas']):
        print(f'Dice for {label}: {dice_all_mean[idx]}')

