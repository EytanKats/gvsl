import os
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

from monai import transforms
from monai.data import DataLoader
from monai.data import Dataset as MonaiDataset


import torch
from torch.utils.data import Dataset as TorchDataset


def read_json_data_file(
        data_file_path,
        data_dir,
        key='training'
):

    with open(data_file_path) as f:
        json_data = json.load(f)

    tr = []
    json_data_training = json_data[key]
    for d in json_data_training:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(data_dir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(data_dir, d[k]) if len(d[k]) > 0 else d[k]
        tr.append(d)

    return tr


class SslDataset(TorchDataset):

    def __init__(self, settings, data_file_path, data_root_dir):
        super(SslDataset, self).__init__()

        self.settings = settings

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.unlabeled_filenames = read_json_data_file(
            data_file_path=data_file_path,
            data_dir=data_root_dir
        )

    def __getitem__(self, index):
        
        # Load image
        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        img = nib.load(self.unlabeled_filenames[random_index]['image']).get_fdata()

        # Resample image, normalize between 0 and 1, add channel axis and convert to float32
        img = zoom(img, (192. / img.shape[0], 128. / img.shape[1], 96. / img.shape[2]), order=1)
        img = img - np.min(img)
        img = img / np.max(img)
        img = img[np.newaxis, :, :, :]
        img = img.astype(np.float32)

        return img

    def __len__(self):
        return self.settings['dataset']['length']


def get_val_transform():

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.AsDiscreted(keys=["label"], to_onehot=13),

            transforms.Spacingd(keys=["image", "label"], pixdim=(3, 3, 3), mode=("bilinear", "nearest")),
            transforms.ScaleIntensityd(keys=["image"], minv=0, maxv=1),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
            transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128)),

            transforms.ToTensord(keys=["image", "label"])
        ]
    )

    return val_transform


def get_data_loaders(settings):

    # Create validation data loader
    val_files = read_json_data_file(
        data_file_path='/mnt/share/data/amos/dataset_preprocessed_mri/annotations/sup_train_24_f0.json',
        data_dir='/mnt/share/data/amos/',
        key='validation'
    )

    val_ds = MonaiDataset(data=val_files, transform=get_val_transform())
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create training data loader
    train_ds = SslDataset(
        settings=settings,
        data_file_path=settings['dataset']['data_file_path'],
        data_root_dir=settings['dataset']['data_dir']
    )
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=settings['dataloader']['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=settings['dataloader']['num_workers'],
        pin_memory=True
    )

    return [train_loader], [val_loader]
