from model import *
from data import *

IS_DATA_AUG = False
USE_MODEL = "last"

config = {
    "validation_steps": 50,
    "steps_epoch": 1000,
    "epochs": 10,
}

if IS_DATA_AUG:
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
else:
    data_gen_args = None

trainGene = trainGenerator(2, '/home/zhaiyu/Dataset/WHU Building Dataset/train',
                           'images', 'masks', data_gen_args, save_to_dir=None)

valGene = valGenerator(2, "/home/zhaiyu/Dataset/WHU Building Dataset/val",
                       'images', "masks", data_gen_args, save_to_dir=None)

model = unet()
model_checkpoint = ModelCheckpoint('models/unet_buildings_weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='loss',
                                   verbose=1, save_best_only=False)

# TODO: continue training from different checkpoint
if USE_MODEL == "last":
    # Load the last model you trained and continue training
    model.load_weights("models/unet_buildings_weights.14-0.04.hdf5", by_name=True)
model.fit_generator(trainGene, validation_data=valGene, validation_steps=config["validation_steps"],
                    steps_per_epoch=config["steps_epoch"], epochs=config["epochs"], callbacks=[model_checkpoint])

# # Test trained model
# testGene = testGenerator("data/whu/test", as_gray=False, flag_multi_class=True)
# results = model.predict_generator(testGene, 100, verbose=1)
# saveResult("data/whu/result", results)
