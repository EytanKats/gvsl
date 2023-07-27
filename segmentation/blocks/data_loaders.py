from monai import transforms
from monai.data import CacheDataset, DataLoader

from segmentation.core.data_utils import read_json_data_file


def get_train_transform(settings):

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),

            transforms.Spacingd(keys=["image", "label"], pixdim=(3, 3, 3), mode=("bilinear", "nearest")),
            transforms.ScaleIntensityd(keys=["image"], minv=0, maxv=1),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=settings['model']['img_size']),
            transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=settings['model']['img_size']),

            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    return train_transform

def get_val_transform(settings):

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),

            transforms.Spacingd(keys=["image", "label"], pixdim=(3, 3, 3), mode=("bilinear", "nearest")),
            transforms.ScaleIntensityd(keys=["image"], minv=0, maxv=1),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=settings['model']['img_size']),
            transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=settings['model']['img_size']),

            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    return val_transform


def get_data_loaders(settings):

    # Get list of training and validation files
    train_files_list = []
    val_files_list = []
    for data_file in settings['dataset']['data_file_paths']:
        train_files, val_files = read_json_data_file(
            data_file_path=data_file,
            data_dir=settings['dataset']['data_dir']
        )

        train_files_list.append(train_files)
        val_files_list.append(val_files)

    # Instantiate training data loaders
    train_loaders_list = []
    for train_files in train_files_list:
        train_transform = get_train_transform(settings)
        train_ds = CacheDataset(data=train_files, transform=train_transform)
        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=settings['dataloader']['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=settings['dataloader']['num_workers'],
            pin_memory=True
        )

        train_loaders_list.append(train_loader)

    # Instantiate validation data loaders
    val_loaders_list = []
    for val_files in val_files_list:
        val_transform = get_val_transform(settings)
        val_ds = CacheDataset(data=val_files, transform=val_transform)
        val_loader = DataLoader(
            dataset=val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=settings['dataloader']['num_workers'],
            pin_memory=True
        )

        val_loaders_list.append(val_loader)

    return train_loaders_list, val_loaders_list


def get_test_data_loaders(settings):
    pass
