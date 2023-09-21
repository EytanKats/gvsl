import os

import torch
import numpy as np

from monai.data import Dataset, DataLoader

from exploration.gvsl import GVSL
from ssl_app.core.plots import plot_tsne
from ssl_app.core.unet3d_gvsl import UNet3D_GVSL
from ssl_app.blocks.data_loaders import read_json_data_file, get_val_transform


OUTPUT_DIR = '/mnt/share/experiments/label/gvsl/tsne/'

pretrained_weights = [
    # '',
    # '/mnt/share/experiments/label/gvsl/original/GVSL_epoch_1000.pth',
    # '/mnt/share/experiments/label/gvsl/sc_nako1000_pretraining_restoration/0/checkpoint/ckpt-49.pth',
    # '/mnt/share/experiments/label/gvsl/sc_nako1000_pretraining_registration/0/checkpoint/ckpt-0.pth',
    # '/mnt/share/experiments/label/gvsl/sc_nako1000_pretraining_registration/0/checkpoint/ckpt-49.pth',
    # '/mnt/share/experiments/label/gvsl/sc_nako1000_pretraining_registration_restoration/0/checkpoint/ckpt-0.pth',
    # '/mnt/share/experiments/label/gvsl/sc_nako1000_pretraining_registration_restoration/0/checkpoint/ckpt-49.pth',
    # '/mnt/share/experiments/label/gvsl/sc_nako1000_pretraining_registration_constrained/0/checkpoint/ckpt-49.pth',
    # '/mnt/share/experiments/label/gvsl/sc_nako1000_pretraining_coupled_convex/0/checkpoint/ckpt-49.pth',
    '/mnt/share/experiments/label/gvsl/sc_nako1000_pretraining_global_local/0/checkpoint/ckpt-44.pth'
]

plot_names = [
    # 'random_initialization',
    # 'gvsl_paper_released_weights',
    # 'restoration_only(models_genesis_augmentations)',
    # 'registration_only_1000steps(small_synthetic_elastic_deformations)',
    # 'registration_only_50000steps(small_synthetic_elastic_deformations)',
    # 'registration_restoration_1000steps',
    # 'registration_restoration_50000steps',
    # 'registration_constrained_50000steps',
    # 'coupled_convex_50000steps',
    'global_local_50000steps'
]


labels = {
    'spleen': 1,
    'right_kidney': 2,
    'left_kidney': 3,
    'gallbladder': 4,
    'esophagus': 5,
    'liver': 6,
    'stomach': 7,
    'aorta': 8,
    'inferior_vena_cava': 9,
    'pancreas': 10,
    'right_adrenal_gland': 11,
    'left_adrenal_gland': 12
}

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create data loader
val_files = read_json_data_file(
    data_file_path='/mnt/share/data/amos/dataset_preprocessed_mri/annotations/sup_train_24_f0.json',
    data_dir='/mnt/share/data/amos/',
    key='validation'
)

val_ds = Dataset(data=val_files, transform=get_val_transform())
val_loader = DataLoader(
    dataset=val_ds,
    batch_size=2,
    shuffle=False,
    num_workers=1,
    pin_memory=True
)

for data_idx, data in enumerate(val_loader):

    # Get image and label
    image = data['image'][0].unsqueeze(0)
    label = data['label'][0].unsqueeze(0)

    image_2 = data['image'][1].unsqueeze(0)
    label_2 = data['label'][1].unsqueeze(0)


    # Send image to device
    image = image.to(device)
    image_2 = image_2.to(device)
    # image_rot90 = torch.rot90(torch.clone(image[0, 0])).unsqueeze(0).unsqueeze(0)

    # Iterate over pretrained weights
    label = label.numpy()[0]
    label_2 = label_2.numpy()[0]
    for plot_name, weights in zip(plot_names, pretrained_weights):

        # Create model
        if 'GVSL_' in weights:
            gvsl = GVSL()
            state_dict = torch.load(weights)
            gvsl.load_state_dict(state_dict)
            model = gvsl.unet
        elif 'ckpt-' in weights:
            model = UNet3D_GVSL(
                n_channels=1,
                chs=(32, 64, 128, 256, 512, 256, 128, 64, 32)
            )
            state_dict = torch.load(weights)
            model.load_state_dict(state_dict['feature_extractor'])
        else:
            model = UNet3D_GVSL(
                n_channels=1,
                chs=(32, 64, 128, 256, 512, 256, 128, 64, 32)
            )

        model.to(device)
        model.eval()

        # Extract features
        _, features = model(image)
        features = features.detach().cpu().numpy()[0]

        _, features_rot90 = model(image_2)
        features_rot90 = features_rot90.detach().cpu().numpy()[0]

        features_list = []
        features_rot90_list = []
        labels_list = []
        labels_rot90_list = []
        for key, value in labels.items():
            class_features = np.transpose(features[:, label[value, ...] == 1])
            class_features_rot90 = np.transpose(features_rot90[:, label_2[value, ...] == 1])

            # if class_features.shape[0] > 1000:
            #     sampled_locations = np.random.choice(np.sum(label[value, ...] == 1), 1000, replace=False)
            #     class_features = class_features[sampled_locations, :]
            #
            # if class_features_rot90.shape[0] > 1000:
            #     sampled_locations_rot90 = np.random.choice(np.sum(label_2[value, ...] == 1), 1000, replace=False)
            #     class_features_rot90 = class_features_rot90[sampled_locations_rot90, :]

            features_list.append(class_features)
            features_rot90_list.append(class_features_rot90)
            labels_list.append([key] * class_features.shape[0])
            labels_rot90_list.append([key + '_2'] * class_features_rot90.shape[0])

        # Show / save TSNE plot
        plot_tsne(
            np.concatenate([features_list[0], features_list[1], features_list[2], features_list[5], features_list[6],
                            features_rot90_list[0], features_rot90_list[1], features_rot90_list[2], features_rot90_list[5], features_rot90_list[6]]),
            np.concatenate([labels_list[0], labels_list[1], labels_list[2], labels_list[5], labels_list[6],
                            labels_rot90_list[0], labels_rot90_list[1], labels_rot90_list[2], labels_rot90_list[5], labels_rot90_list[6]]),
            num_classes=10,  #len(labels),
            title=plot_name,
            output_path=os.path.join(OUTPUT_DIR, f'{plot_name}_im{data_idx}_2images.png'),
            show=True)

        plot_tsne(
            np.concatenate(features_list),
            np.concatenate(labels_list),
            num_classes=len(labels),
            title=plot_name,
            output_path=os.path.join(OUTPUT_DIR, f'{plot_name}_single_image_{data_idx}.png'),
            show=True)





