from model import *
from data import *

model = unet()

model.load_weights("models/unet_buildings_weights.14-0.04.hdf5", by_name=True)

testGene = testGenerator("data/whu/test", as_gray=False, flag_multi_class=True)
results = model.predict_generator(testGene,100,verbose=1)
saveResult("data/whu/result",results)
