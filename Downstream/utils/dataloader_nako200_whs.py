import os
import json
from os.path import join
from os import listdir
import SimpleITK as sitk
from torch.utils import data
import numpy as np
from scipy.ndimage import zoom
from matplotlib import pyplot as plt

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz"])

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

    val = []
    json_data_validation = json_data['validation']
    for d in json_data_validation:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(data_dir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(data_dir, d[k]) if len(d[k]) > 0 else d[k]
        val.append(d)

    return tr, val

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, mode, num_classes, shape=(192, 192, 192)):
        super(DatasetFromFolder3D, self).__init__()

        train_files, val_files = read_json_data_file(
            # data_file_path='/mnt/share/data/nako/annotations/niiregion2_demopartial_wmod_fold2_tr16.json',
            data_file_path='/share/data_supergrover3/kats/data/amos/dataset_preprocessed_ct/annotations/sup_train_24_f0.json',
            # data_dir='/mnt/share/data/nako',
            data_dir='/share/data_supergrover3/kats/data/amos/'
        )

        if mode == 'train':
            # np.random.seed(2019)
            # self.files = np.random.choice(train_files, size=40, replace=False).tolist()
            self.files = train_files
        else:
            # np.random.seed(2019)
            # self.files = np.random.choice(val_files, size=20, replace=False).tolist()
            self.files = val_files

        # self.labeled_filenames = [x for x in listdir(join(labeled_file_dir, 'image')) if is_image_file(x)]
        # self.labeled_file_dir = labeled_file_dir

        self.num_classes = num_classes
        self.shape = shape

    def __getitem__(self, index):
        img = sitk.ReadImage(self.files[index]['image'])
        img = sitk.GetArrayFromImage(img)

        img = np.swapaxes(img, 0, 2)
        img = zoom(img, (128. / img.shape[0], 128. / img.shape[1], 128. / img.shape[2]), order=1)

        # vol_q_1 = np.percentile(img, 1)
        # vol_q_99 = np.percentile(img, 99.99)
        # vol_mean = np.mean(img)
        # vol_std = np.std(img)
        # img = np.clip(img, vol_q_1, vol_q_99)
        # img = (img - vol_mean) / max(vol_std, 1e-8)

        img = img - np.min(img)
        img = img / np.max(img)
        img = img.astype(np.float32)

        # center_slice = img.shape[2] // 2
        # for image_slice in range(center_slice - 2, center_slice + 3):
        #     plt.imshow(img[:, :, image_slice], cmap='gray')
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()

        img = img[np.newaxis, :, :, :]

        lab = sitk.ReadImage(self.files[index]['label'])
        lab = sitk.GetArrayFromImage(lab)
        lab = np.swapaxes(lab, 0, 2)

        lab = zoom(lab, (128. / lab.shape[0], 128. / lab.shape[1], 128. / lab.shape[2]), order=0)

        # center_slice = lab.shape[2] // 2
        # for image_slice in range(center_slice - 2, center_slice + 3):
        #     plt.imshow(lab[:, :, image_slice])
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()

        lab = self.to_categorical(lab, self.num_classes)
        lab = lab.astype(np.float32)

        return img, lab, self.files[index]['image']

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((num_classes, n))
        categorical[y, np.arange(n)] = 1
        output_shape = (num_classes,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __len__(self):
        return len(self.files)

