import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from torch.utils import data
from scipy.ndimage import zoom


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

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, data_file_path, data_root_dir):
        super(DatasetFromFolder3D, self).__init__()

        self.unlabeled_filenames = read_json_data_file(
            data_file_path=data_file_path,
            data_dir=data_root_dir
        )

    def __getitem__(self, index):
        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        unlabeled_img1 = nib.load(self.unlabeled_filenames[random_index]['image']).get_fdata()

        unlabeled_img1 = zoom(unlabeled_img1, (128. / unlabeled_img1.shape[0], 128. / unlabeled_img1.shape[1], 128. / unlabeled_img1.shape[2]), order=1)
        unlabeled_img1 = unlabeled_img1 - np.min(unlabeled_img1)
        unlabeled_img1 = unlabeled_img1 / np.max(unlabeled_img1)

        unlabeled_img1 = unlabeled_img1.astype(np.float32)

        # center_slice = unlabeled_img1.shape[2] // 2
        # for image_slice in range(center_slice - 2, center_slice + 3):
        #     plt.imshow(unlabeled_img1[:, :, image_slice], cmap='gray')
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()

        unlabeled_img1 = unlabeled_img1[np.newaxis, :, :, :]

        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        unlabeled_img2 = nib.load(self.unlabeled_filenames[random_index]['image']).get_fdata()

        unlabeled_img2 = zoom(unlabeled_img2, (128. / unlabeled_img2.shape[0], 128. / unlabeled_img2.shape[1], 128. / unlabeled_img2.shape[2]), order=1)
        unlabeled_img2 = unlabeled_img2 - np.min(unlabeled_img2)
        unlabeled_img2 = unlabeled_img2 / np.max(unlabeled_img2)

        unlabeled_img2 = unlabeled_img2.astype(np.float32)

        # center_slice = unlabeled_img2.shape[2] // 2
        # for image_slice in range(center_slice - 2, center_slice + 3):
        #     plt.imshow(unlabeled_img2[:, :, image_slice], cmap='gray')
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()

        unlabeled_img2 = unlabeled_img2[np.newaxis, :, :, :]

        return unlabeled_img1, unlabeled_img2

    def __len__(self):
        return len(self.unlabeled_filenames)

