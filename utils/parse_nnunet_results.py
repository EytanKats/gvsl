import os
import json

import numpy as np
import pandas as pd


RESULTS_DIR = '/share/data_supergrover3/kats/nnunet/nnUNet_results/Dataset804_AMOSCT30/nnUNetTrainer_gvsl_sslnako1000ckpt1000_ct_amos24__nnUNetPlans__3d_fullres'
SUMMARY_FILE_RELATIVE_PATH = 'validation/summary.json'
OUTPUT_FILE_NAME = 'summary.csv'
FOLDS = [0, 1, 2]

LABELS = {
    '1': "spleen",
    '2': "kidney_right",
    '3': "kidney_left",
    '4': "gallbladder",
    '5': "esophagus",
    '6': "liver",
    '7': "stomach",
    '8': "aorta",
    '9': "inferior_vena_cava",
    '10': "pancreas",
    '11': "adrenal_gland_right",
    '12': "adrenal_gland_left"
}

json_data = []
for fold in FOLDS:
    summary_file_path = os.path.join(RESULTS_DIR, f'fold_{fold}', SUMMARY_FILE_RELATIVE_PATH)

    with open(summary_file_path) as f:
        json_data.append(json.load(f))

df = pd.DataFrame()
for key, value in LABELS.items():
    dice = []
    for fold in FOLDS:
        dice.append(json_data[fold]['mean'][key]['Dice'])

    dice.append(np.mean(dice))
    dice.append(np.std(dice))
    df[value] = dice

dice = []
for fold in FOLDS:
    dice.append(json_data[fold]['foreground_mean']['Dice'])

dice.append(np.mean(dice))
dice.append(np.std(dice))
df['mean'] = dice

output_file_path = os.path.join(RESULTS_DIR, OUTPUT_FILE_NAME)
df.to_csv(output_file_path, index=False)
