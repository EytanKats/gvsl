import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

from create_data_file import create_json_data_file

ROOT_DIR = '/share/data_supergrover3/kats/data/amos'
DATA_DIR = '/share/data_supergrover3/kats/data/amos/dataset_preprocessed_ct'
NUM_SSL_FILES = 240
NUM_SUP_FILES = 30
NUM_FOLDS = 5
NUM_SUP_TRAIN_FILES = [24, 16, 8, 4]

OUTPUT_SSL_FILE_PATH = '/share/data_supergrover3/kats/data/amos/dataset_preprocessed_ct/annotations/ssl_240.json'
OUTPUT_SUP_TEST_FILE_PATH = '/share/data_supergrover3/kats/data/amos/dataset_preprocessed_ct/annotations/sup_test_30.json'
OUTPUT_SUP_TRAIN_FILE_PATH_TEMPLATE = '/share/data_supergrover3/kats/data/amos/dataset_preprocessed_ct/annotations/sup_train'

# Get list of relevant files
files = []
for path in Path(DATA_DIR).rglob('images*/*.nii.gz'):
    files.append(str(path.relative_to(ROOT_DIR)))
files = np.array(sorted(files))

masks = []
for path in Path(DATA_DIR).rglob('labels*/*.nii.gz'):
    masks.append(str(path.relative_to(ROOT_DIR)))
masks = np.array(sorted(masks))

# Choose files for SSL training, supervised training and hold-out test files
indices = np.arange(len(files))
np.random.shuffle(indices)

ssl_indices = indices[:NUM_SSL_FILES]
sup_train_indices = indices[NUM_SSL_FILES:NUM_SSL_FILES + NUM_SUP_FILES]
sup_test_indices = indices[NUM_SSL_FILES + NUM_SUP_FILES:]

ssl_files = files[ssl_indices]
create_json_data_file(ssl_files, OUTPUT_SSL_FILE_PATH)

sup_test_files = files[sup_test_indices]
sup_test_masks = masks[sup_test_indices]
create_json_data_file(sup_test_files, OUTPUT_SUP_TEST_FILE_PATH, train_masks=sup_test_masks, train_key='test')

sup_files = files[sup_train_indices]
sup_masks = masks[sup_train_indices]

# Split files for supervised to training and validation
kf = KFold(n_splits=NUM_FOLDS, shuffle=True)
for fold, (train_idx, val_idx) in enumerate(kf.split(sup_files)):

    train_idx = train_idx.tolist()
    val_idx = val_idx.tolist()

    val_files = sup_files[val_idx]
    val_masks = sup_masks[val_idx]
    for num_subset_files in NUM_SUP_TRAIN_FILES:
        train_subset_idx = np.random.choice(train_idx, num_subset_files, replace=False)
        train_files = sup_files[train_subset_idx]
        train_masks = sup_masks[train_subset_idx]

        output_file_path = f'{OUTPUT_SUP_TRAIN_FILE_PATH_TEMPLATE}_{num_subset_files}_f{fold}.json'
        create_json_data_file(train_files, output_file_path, train_masks=train_masks, val_files=val_files, val_masks=val_masks)
