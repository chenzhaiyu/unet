train_config = {
    "validation_steps": 50,
    "steps_epoch": 1000,
    "epochs": 20,
    "batch_size": 2,
    "weights_dir": "models",
    "train_data_dir": "D:/Datasets/WHU Building Dataset/train",
    "val_data_dir": "D:/Datasets/WHU Building Dataset/val"
}

test_config = {
    "weights_path": "models/unet_buildings_weights.14-0.04.hdf5",
    "test_data_dir": "D:/Datasets/WHU Building Dataset/test",
    "save_dir": "data/whu/result"
}

data_aug_config = dict(rotation_range=0.2,
                       width_shift_range=0.05,
                       height_shift_range=0.05,
                       shear_range=0.05,
                       zoom_range=0.05,
                       horizontal_flip=True,
                       fill_mode='nearest')
