import os
import json
from monai import transforms
from monai.data import Dataset, CacheDataset, DataLoader

from segmentation.core.data_utils import read_json_data_file


def get_train_transform():

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.AsDiscreted(keys=["label"], to_onehot=13),

            transforms.Spacingd(keys=["image", "label"], pixdim=(3, 3, 3), mode=("bilinear", "nearest")),
            transforms.ScaleIntensityd(keys=["image"], minv=0, maxv=1),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
            transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128)),

            # transforms.RandRotated(keys=["image", "label"], range_x=15, range_y=15, range_z=15, prob=0.4),
            # transforms.RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.7, max_zoom=1.4),
            #
            # transforms.RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.1),
            # transforms.RandGaussianSmoothd(keys="image", prob=0.2, sigma_x=(0.5, 1), sigma_y=(0.5, 1),
            #                                sigma_z=(0.5, 1)),
            #
            # transforms.RandScaleIntensityd(keys="image", prob=0.15, factors=0.25),
            # transforms.RandShiftIntensityd(keys="image", prob=0.15, offsets=0.25),
            # transforms.RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.7, 1.5)),

            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    return train_transform


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


            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    return val_transform


def get_data_loaders():

    train_files, val_files = read_json_data_file(
        data_file_path='/mnt/share/data/amos/dataset_preprocessed_mri/annotations/sup_train_24_f0.json',
        data_dir='/mnt/share/data/amos/'
    )

    with open('/mnt/share/data/amos/dataset_preprocessed_mri/annotations/sup_test_30.json') as f:
        json_data = json.load(f)

    test_files = []
    json_data_training = json_data['test']
    for d in json_data_training:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join('/share/data_supergrover3/kats/data/amos/', iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join('/share/data_supergrover3/kats/data/amos/', d[k]) if len(d[k]) > 0 else d[k]
        test_files.append(d)

    train_ds = Dataset(data=train_files, transform=get_train_transform())
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    val_ds = Dataset(data=val_files, transform=get_train_transform())
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_ds = Dataset(data=test_files, transform=get_train_transform())
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
