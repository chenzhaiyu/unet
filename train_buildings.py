from model import *
from data import *

# data_gen_args = dict(rotation_range=0.2,
#                      width_shift_range=0.05,
#                      height_shift_range=0.05,
#                      shear_range=0.05,
#                      zoom_range=0.05,
#                      horizontal_flip=True,
#                      fill_mode='nearest')

data_gen_args = None

trainGene = trainGenerator(2, '/home/zhaiyu/Dataset/WHU Building Dataset/train', 'images', 'masks', data_gen_args, save_to_dir=None)
valGene = valGenerator(2, "/home/zhaiyu/Dataset/WHU Building Dataset/val", 'images', "masks", data_gen_args, save_to_dir=None)

model = unet()

model_checkpoint = ModelCheckpoint('models/unet_buildings_weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='loss', verbose=1, save_best_only=False)

# TODO: continue training
init_with = "last"
if init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights("models/unet_buildings_weights.10-0.04.hdf5", by_name=True)
model.fit_generator(trainGene, validation_data=valGene, validation_steps=50, steps_per_epoch=1000, epochs=10, callbacks=[model_checkpoint])

testGene = testGenerator("data/whu/test", as_gray=False, flag_multi_class=True)
results = model.predict_generator(testGene,100,verbose=1)
saveResult("data/whu/result",results)