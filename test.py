from model import *
from data import *
from config import test_config

model = unet()

model.load_weights(test_config["weights_path"], by_name=True)

testGene = testGenerator(test_config["test_data_dir"], as_gray=False, flag_multi_class=True)
results = model.predict_generator(testGene, 100, verbose=1)
saveResult(test_config["save_dir"], results)
