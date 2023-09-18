mlops_settings = {
    'use_mlops': True,
    'project_name': 'Label',
    'task_name': 'sc_nako1000_pretraining_coupled_convex',
    'task_type': 'training',
    'tags': [],
    'connect_arg_parser': False,
    'connect_frameworks': False,
    'resource_monitoring': True,
    'connect_streams': True
}

manager_settings = {
    'output_folder': '/mnt/share/experiments/label/gvsl/sc_nako1000_pretraining_coupled_convex/',
    'active_folds': [0],
    'restore_checkpoint': False,
    'restore_checkpoint_path': ''
}

trainer_settings = {
    'epochs': 500,
    'val_freq': 10,
    'monitor': 'mse',
    'monitor_regime': 'min',
    'ckpt_freq': 10,
    'ckpt_save_best_only': False,
    'use_early_stopping': False,
    'early_stopping_patience': -1,
    'plateau_patience': -1
}

dataset_settings = {
    'data_dir': '/mnt/share/data/nako_1000/',
    'data_file_path': '/mnt/share/data/nako_1000/nii_region2_wcontrast_preprocessed/annotations/ssl_1078.json',

    # 'labels': {
    #     'spleen': 0,
    #     'right_kidney': 1,
    #     'left_kidney': 2,
    #     'gallbladder': 3,
    #     'esophagus': 4,
    #     'liver': 5,
    #     'stomach': 6,
    #     'aorta': 7,
    #     'inferior_vena_cava': 8,
    #     'pancreas': 9,
    #     'right_adrenal_gland': 10,
    #     'left_adrenal_gland': 11
    # }

    'length': 100
}

transforms_settings = {
}

data_loader_settings = {
    'batch_size': 2,
    'num_workers': 4
}

model_settings = {
}

optimizer_settings = {
    'lr': 1e-4
}

scheduler_settings = {
}

loss_functions_settings = {
    'restoration_loss_name': 'restoration_mse',
    'registration_loss_name': 'registration_ncc',
    'flow_regularization_loss_name': 'flow_regularization_gradient',
    'weight_restoration': 1,
    'weight_registration': 1,
    'weight_flow_regularization': 0
}

metrics_settings = {
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
    'scheduler': scheduler_settings,
    'loss': loss_functions_settings,
    'metrics': metrics_settings,
    'app': app_settings,
    'postprocessor': postprocessor_settings
}
