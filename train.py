from model import *
from data import *
from config import train_config, data_aug_config

IS_DATA_AUG = False
MODEL_NAME = None  # can be chosen from [None, "model_filename", "last"]
MODEL_DIR = "models"


if IS_DATA_AUG:
    data_gen_args = data_aug_config
else:
    data_gen_args = None

trainGene = trainGenerator(train_config["batch_size"], train_config["train_data_path"],
                           'images', 'masks', data_gen_args, save_to_dir=None)

valGene = valGenerator(train_config["batch_size"], train_config["val_data_path"],
                       'images', "masks", data_gen_args, save_to_dir=None)

model = unet()
model_checkpoint = ModelCheckpoint('models/unet_buildings_weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='loss',
                                   verbose=1, save_best_only=False)

initial_epoch = 0

if MODEL_NAME == "last":
    # Load the last model you trained and continue training
    last_model_path, initial_epoch = find_last(MODEL_DIR)
    model.load_weights(last_model_path, by_name=True)

elif isinstance(MODEL_NAME, str) and MODEL_NAME != "last":
    print("loading weights from {}".format(MODEL_NAME))
    model.load_weights(os.path.join(MODEL_DIR, MODEL_NAME), by_name=True)

elif MODEL_NAME is None:
    print("training model from scratch")

else:
    raise NotImplementedError

model.fit_generator(trainGene, validation_data=valGene, validation_steps=train_config["validation_steps"],
                    steps_per_epoch=train_config["steps_epoch"], epochs=train_config["epochs"],
                    callbacks=[model_checkpoint], initial_epoch=initial_epoch)

