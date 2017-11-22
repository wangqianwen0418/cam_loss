import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import keras.backend as K
from keras.models import load_model

import requests
import io
import cv2
from PIL import Image as Image_PIL
import numpy as np
import os
IMG_NAME = "003.png"
DIR = "/home/qianww/hat_mask/data/"
IMG_PATH = os.path.join(DIR, IMG_NAME) 
MODEL_PATH = "/home/qianww/hat_mask/resnet56/save_model/resnet_model.h5"

def download_img(url, name="test"):
    response = requests.get(url)
    img_pil = Image_PIL.open(io.BytesIO(response.content))
    fname = '{}.jpg'.format(name)
    img_pil.save(fname)
    return fname

def get_cam(features, weights, idx):
    size = (224, 224)
    bs, h, w, c = features.shape
    features = features.reshape((h*w, c))
    cam =  np.matmul(features, weights[:, idx])
    cam = cam.reshape(h, w)
    # normalize for img display
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam) 
    cam_img = np.uint8(255 * cam_img)
    cam = cv2.resize(cam_img, size)
    return cam

def heatmap(cam, img_path='', save_name="cam_keras"):
    if img_path == '':
        (h, w) = (224, 224)
        img = np.zeros(size)
    else:
        img = cv2.imread(img_path)
        (h, w, _) = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(cam, (w, h)), cv2.COLORMAP_JET)
    result = heatmap*0.5 + img * 0.5
    cv2.imwrite('{}.jpg'.format(save_name), result)
    cv2.waitKey(500)


# model = ResNet50(weights='imagenet')
# img_path = download_img(IMG_URL)
model = load_model(MODEL_PATH)
img_path = IMG_PATH

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

last_conv_name = "activation_49"
last_conv = model.get_layer(name=last_conv_name)
func = K.function([model.input, K.learning_phase()], [last_conv.output])
feature_conv = func([x, 0])[0] # learning phase = 0 in test mode

# fc = "fc1000"
fc = "global_average_pooling2d_1 "
weights = model.get_layer(name="dense_1").get_weights()[0]

# preds = model.predict(x)
# model.summary()
# top_n = 3
# idxs = preds[0].argsort()[::-1][:top_n]


top_n=3
preds = model.predict(x)[0]
labels = ["no person", "person without hat", "everyone has hat"]
idxs = preds.argsort()[::-1][:top_n]
print([(preds[i], labels[i]) for i in idxs])

for idx in idxs:
    cam = get_cam(feature_conv, weights, idx)
    heatmap(cam, img_path=img_path, save_name="imgs/kitchen/{}_{}".format(IMG_NAME, labels[idx]))
