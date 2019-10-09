train_config = {
    "validation_steps": 50,
    "steps_epoch": 1000,
    "epochs": 20,
    "batch_size": 2,
}

data_aug_config = dict(rotation_range=0.2,
                       width_shift_range=0.05,
                       height_shift_range=0.05,
                       shear_range=0.05,
                       zoom_range=0.05,
                       horizontal_flip=True,
                       fill_mode='nearest')
