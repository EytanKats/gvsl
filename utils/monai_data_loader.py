import os
import json
from monai import transforms
from monai.data import Dataset, DataLoader


def read_json_data_file(
        data_file_path,
        data_dir
):

    with open(data_file_path) as f:
        json_data = json.load(f)

    tr = []
    json_data_training = json_data['training']
    for d in json_data_training:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(data_dir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(data_dir, d[k]) if len(d[k]) > 0 else d[k]
        tr.append(d)

    return tr


def get_train_transform():

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.AddChanneld(keys=["image"]),

            transforms.Spacingd(keys=["image"], pixdim=(3, 3, 3), mode=("bilinear")),
            transforms.ScaleIntensityd(keys=["image"], minv=0, maxv=1),
            transforms.SpatialPadd(keys=["image"], spatial_size=(192, 128, 96)),
            transforms.CenterSpatialCropd(keys=["image"], roi_size=(192, 128, 96)),

            transforms.ToTensord(keys=["image"]),
        ]
    )

    return train_transform


def get_data_loader(data_file_path, data_root_dir):

    train_files = read_json_data_file(
        data_file_path=data_file_path,
        data_dir=data_root_dir
    )

    train_ds = Dataset(data=train_files, transform=get_train_transform())

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    return train_loader
