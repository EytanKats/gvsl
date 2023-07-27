mlops_settings = {
    'use_mlops': True,
    'project_name': 'Label',
    'task_name': 'gvsl_randominit_mri_amos24',
    'task_type': 'training',
    'tags': ['gvsl', 'randominit', 'mri', 'amos24'],
    'connect_arg_parser': False,
    'connect_frameworks': False,
    'resource_monitoring': True,
    'connect_streams': True
}

manager_settings = {
    'output_folder': '/share/data_supergrover3/kats/experiments/label/gvsl/gvsl_randominit_mri_amos24/',
    'active_folds': [0, 1, 2, 3, 4],
    'restore_checkpoint': False,
    'restore_checkpoint_path': ''
}

trainer_settings = {
    'epochs': 200,
    'val_freq': 1,
    'monitor': 'dice_mean',
    'monitor_regime': 'max',
    'ckpt_freq': 1,
    'ckpt_save_best_only': True,
    'use_early_stopping': False,
    'early_stopping_patience': -1,
    'plateau_patience': -1
}

dataset_settings = {
    'data_dir': '/share/data_supergrover3/kats/data/amos/',
    'data_file_paths': [
        '/share/data_supergrover3/kats/data/amos/dataset_preprocessed_mri/annotations/sup_train_24_f0.json',
        '/share/data_supergrover3/kats/data/amos/dataset_preprocessed_mri/annotations/sup_train_24_f1.json',
        '/share/data_supergrover3/kats/data/amos/dataset_preprocessed_mri/annotations/sup_train_24_f2.json',
        '/share/data_supergrover3/kats/data/amos/dataset_preprocessed_mri/annotations/sup_train_24_f3.json',
        '/share/data_supergrover3/kats/data/amos/dataset_preprocessed_mri/annotations/sup_train_24_f4.json'
    ],

    'labels': {
        'spleen': 0,
        'right_kidney': 1,
        'left_kidney': 2,
        'gallblader': 3,
        'esophagus': 4,
        'liver': 5,
        'stomach': 6,
        'aorta': 7,
        'inferior_vena_cava': 8,
        'pancreas': 9,
        'right_adrenal_gland': 10,
        'left_adrenal_gland': 11
    }
}

transforms_settings = {

}

data_loader_settings = {
    'batch_size': 1,
    'num_workers': 4
}

model_settings = {
    'architecture': 'unet3d_gvsl_finetune',
    'pretrained_weights': None,
    'img_size': (128, 128, 128),
    'out_channels': 14,
    'val_roi_size': (96, 96, 96),
    'val_sw_batch_size': 1,
    'overlap': 0.5
}

optimizer_settings = {
    'lr': 1e-4
}

loss_functions_settings = {
    'loss_name': 'dice',
    'include_background': True,
    'to_onehot_y': True,
    'sigmoid': False,
    'softmax': True
}

metrics_settings = {
    'metric_name': 'dice',
    'include_background': False
}

app_settings = {
    'use_reduce_lr_on_plateau': False,
    'reduce_lr_on_plateau_factor': 0.9,
    'reduce_lr_on_plateau_min': 1e-6,
    'use_ema': False,
    'ema_decay': -1
}

postprocessor_settings = {
}

settings = {
    'mlops': mlops_settings,
    'manager': manager_settings,
    'trainer': trainer_settings,
    'dataset': dataset_settings,
    'transforms': transforms_settings,
    'dataloader': data_loader_settings,
    'model': model_settings,
    'optimizer': optimizer_settings,
    'loss': loss_functions_settings,
    'metrics': metrics_settings,
    'app': app_settings,
    'postprocessor': postprocessor_settings
}
