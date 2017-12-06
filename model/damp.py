import keras
from keras.models import load_model
from keras.layers import Dense
import keras.backend as K

from get_coco import COCOData
from model import FC_layer
from losses import area_loss

import numpy as np
import os
import json

keras.losses.area_loss = area_loss

# #data
save_path = "../data/cache"
batch_size=32
target_size=(224, 224)
cocodata_train = COCOData(data_dir="../data/coco/", COI=['cat'], img_set="train", year=2017, batch_size=batch_size, target_size=target_size)
x_train,[y_train, bbox_train] = cocodata_train.get_data()

# model_path = "save_model/vgg19_exp1.h5"
# model = load_model(model_path, custom_objects={'FC_layer': FC_layer})

# preds = model.predict(x_train, batch_size=batch_size)
# # score = model.evaluate(x_train, y_train, batch_size=batch_size)

# conv5 = "block5_pool"
# conv5_features = model.get_layer(name=conv5).output
# func = K.function([model.input]+[K.learning_phase()],[conv5_features])

# for bs in range(x_train.shape[0]//batch_size):
#     maps = np.array(func([x_train[bs*batch_size:(bs+1)*batch_size], 0.]))[0]
#     if bs==0:
#         allmaps = maps
#     else:
#         allmaps = np.concatenate((allmaps, maps), axis=0)

# # results = {"allmaps": list(allmaps), "preds": list(preds[0]), "pred_maps": list(preds[1])}

# # with open(os.path.join(save_path, "results.json"), 'w') as jsonf:
# #     json.dump(results,jsonf)
# # jsonf.close()

# np.savez(os.path.join(save_path, "results.npz"), allmaps=allmaps, preds=preds[0], pred_maps=preds[1])
# print('all maps shape', allmaps.shape, "preds shape", preds[0].shape, preds[1].shape)

model_path = "save_model/vgg19_cat_nobbox.h5"
model = load_model(model_path, custom_objects={'FC_layer': FC_layer})

preds = model.predict(x_train, batch_size=batch_size)

conv5 = "block5_pool"
conv5_features = model.get_layer(name=conv5).output
func_maps = K.function([model.input]+[K.learning_phase()],[conv5_features])

fc="last_fc"
weights = model.get_layer(name=fc).get_weights()[0]
for bs in range(x_train.shape[0]//batch_size):
    maps = np.array(func_maps([x_train[bs*batch_size:(bs+1)*batch_size], 0.]))[0]
    pred_map = np.matmul(maps, weights)
    if bs==0:
        allmaps = maps
        pred_maps = pred_map
    else:
        allmaps = np.concatenate((allmaps, maps), axis=0)
        pred_maps = np.concatenate((pred_maps, pred_map), axis=0)

np.savez(os.path.join(save_path, "results_nobbox.npz"), allmaps=allmaps, preds=preds, pred_maps=pred_maps)
print('all maps shape', allmaps.shape, "preds shape", preds.shape, "pred map shape", pred_maps.shape)





