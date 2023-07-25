import numpy as np
from pathlib import Path

from create_data_file import create_json_data_file

ROOT_DIR = '/share/data_supergrover3/kats/data/nako_1000/'
IMAGES_DIR = '/share/data_supergrover3/kats/data/nako_1000/nii_region2_wcontrast_preprocessed'
NUM_TRAIN_FILES = 240
OUTPUT_FILE_PATH = '/share/data_supergrover3/kats/data/nako_1000/nii_region2_wcontrast_preprocessed/annotations/ssl_240.json'

# Get list of relevant files
files = []
for path in Path(IMAGES_DIR).rglob('*.nii.gz'):
    files.append(str(path.relative_to(ROOT_DIR)))
files = np.array(sorted(files))

# Sample files randomly
files = np.random.choice(files, size=NUM_TRAIN_FILES, replace=False).tolist()
create_json_data_file(files, OUTPUT_FILE_PATH)
