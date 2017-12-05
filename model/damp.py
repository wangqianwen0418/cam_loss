import keras
from keras.models import load_model
from keras.layers import Dense
import keras.backend as K

from get_coco import COCOData
from model import FC_layer
from losses import area_loss

import numpy as np

# keras.losses.area_loss = area_loss

# model_path = "save_model/vgg19_exp1.h5"
# model = load_model(model_path, custom_objects={'FC_layer': FC_layer})

#data
batch_size=32
target_size=(224, 224)
cocodata_train = COCOData(data_dir="../data/coco/", COI=['cat'], img_set="train", batch_size=batch_size, target_size=target_size)
# x_train,[y_train, bbox_train] = cocodata_train.get_data(0)

# preds = model.predict(x_train, batch_size=batch_size)
# # score = model.evaluate(x_train, y_train, batch_size=batch_size)

# last_conv = "block5_pool"
# conv_features = model.get_layer(name=last_conv).output
# func = K.function([model.input]+[K.learning_phase()],[conv_features])

# for bs in range(x_train.shape[0]//batch_size):
#     maps = np.array(func([x_train[bs*batch_size:(bs+1)*batch_size], 0.]))[0]
#     if bs==0:
#         all_maps = maps
#     else:
#         all_maps = np.concatenate((all_maps, maps), axis=0)
# print('all maps shape', all_maps.shape, "preds shape", preds[0].shape, preds[1].shape)

cocodata_val = COCOData(data_dir="../data/coco/", COI=['cat'], img_set="val", batch_size=batch_size, target_size=target_size)
print(cocodata_val.num_images, cocodata_train.num_images)
    

