import keras
import keras.utils.np_utils as np_utils
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras import backend as K

import argparse
import numpy as np
import os
from random import shuffle

ratio = 0.8
batch_size = 32
epochs = 80
data_dir = "../data/kitchen"
target = "person_nohat"


# def load_data(data_dir):
#     file_meta = []
#     label_path = os.path.join(data_dir, "labels.txt")
#     with open(label_path, "r") as f:
#         lines = f.read().split("\n") 
#     for line in lines:
#         items = line.split("\t")
#         if len(items) != 5:
#             break
#         file_meta.append(items)
#     data = []
#     labels = []
#     for item in file_meta:
#         fname = os.path.join(data_dir, item[0] + ".png")
#         if not os.path.isfile(fname):
#             fname = os.path.join(data_dir, item[0] + ".jpg")
#         img = image.load_img(fname, target_size=(224, 224))
#         x = image.img_to_array(img)
#         data.append(x)
#         labels.append(list(map(int, item[1:])))
#     data = np.array(data)
#     data = preprocess_input(data)
#     labels = np.array(labels)
#     return data, labels

def load_data(data_dir, mode, target, ratio):
    """
    # Augments:
        data_dir: the path of data.
        ratio: the percent of data that used for traing.
        target: one of ("total", "no_suit", "no_mask", "no_hat").
        mode: one of ("train", "val").
    # Return:
        (data, label)
    """
    w = 224
    h = 224
    imgs = []
    imgs_noperson = []
    labels = []
    targets = ["index", "total", "no_suit", "no_mask", "no_hat", "person_nohat"]
    for idx, file_name in enumerate(sorted(os.listdir(data_dir))):
        f_path = os.path.join(data_dir, file_name)
        if file_name.lower().endswith(( '.jpeg', '.jpg', '.png')):
            img = image.load_img(f_path, target_size=(224, 224))
            x = image.img_to_array(img)
            imgs.append(x)
        elif file_name=="labels.txt":
            with open(f_path,"r") as f:
                for line in f:
                    line = line.split()
                    # print(line)
                    label_i = targets.index(target)
                    if label_i>4:
                        if int(line[1])>0 and int(line[3])>0:
                            label = 1
                        elif int(line[1])>0 and int(line[3])==0:
                            label = 2 
                    else:
                        label = int(line[label_i])>0
                    labels.append(label)
        else:
            for idx, img_name in  enumerate(sorted(os.listdir(f_path))):
                img_path = os.path.join(f_path, img_name)
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                imgs_noperson.append(x)
    imgs = np.copy(imgs)
    imgs = preprocess_input(imgs)
    imgs_noperson = np.copy(imgs_noperson)
    imgs_noperson = preprocess_input(imgs_noperson)
    labels = np.copy(labels)
    num_train_person = int(imgs.shape[0]*ratio)
    num_train_noperson = int(imgs_noperson.shape[0]*ratio)
    if mode=="train":
        x_train = np.concatenate((imgs[:num_train_person], imgs_noperson[:num_train_noperson]), axis=0)
        y_train = np.concatenate((labels[:num_train_person], np.zeros(num_train_noperson,dtype=int)), axis=0)
        # x_train = imgs[:num_train_person]
        # y_train = labels[:num_train_person]
        return x_train, y_train
    else:
        x_val = np.concatenate((imgs[num_train_person:], imgs_noperson[num_train_noperson:]), axis=0)
        y_val = np.concatenate((labels[num_train_person:], np.zeros(int(imgs_noperson.shape[0]*(1-ratio)+1), dtype=int)), axis=0)
        # x_val = imgs[num_train_person:]
        # y_val = labels[num_train_person:]
        return x_val, y_val

parser = argparse.ArgumentParser()
parser.add_argument("--exp", required=True)
args = parser.parse_args()
exp = args.exp

x_train, y_train = load_data(data_dir=data_dir, mode="train", target=target, ratio=ratio)
x_val, y_val = load_data(data_dir=data_dir, mode="val", target=target, ratio=ratio)
# print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
# max_count = max(np.max(y_train), np.max(y_val))+1
max_count = 3
y_train = keras.utils.to_categorical(y_train, max_count)
y_val = keras.utils.to_categorical(y_val, max_count)
## callbacks to save the best model
check_point = keras.callbacks.ModelCheckpoint("save_model/vgg_model_{}.h5".format(exp), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False)
tf_log = keras.callbacks.TensorBoard(log_dir='save_model/tf_logs_{}'.format(exp), batch_size=batch_size, write_graph=True)
callbacks = [check_point, tf_log]

#
datagen = ImageDataGenerator(
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=(0.8, 1),
    horizontal_flip=True)
datagen.fit(x_train)

### for recording experiment 
print("epochs:{}".format(epochs), "target:{}".format(target), "datagen:{}".format(datagen))
# create the base pre-trained model
base_model = VGG19(weights='imagenet', include_top=False, input_shape = (224,224,3), pooling="avg") ## global average pooling has the best performance
x = base_model.output
#x = Dense(256, activation = 'relu')(x)
# x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(max_count, activation='softmax')(x)
#nosuit_output = Dense(256, activation='relu')(x)
#nosuit_output = Dense(max_count + 1, activation='softmax')(nosuit_output)
#nomask_output = Dense(256, activation='relu')(x)
#nomask_output = Dense(max_count + 1, activation='softmax')(nomask_output)
# nohat_output = Dense(256, activation='relu')(x)
# nohat_output = Dense(max_count + 1, activation='softmax')(nohat_output)

#model = Model(inputs= base_model.input, outputs=[total_output, nosuit_output, nomask_output, nohat_output])

model = Model(inputs= base_model.input, outputs=x)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['accuracy'])
print("stage 1")
model.summary()
# train the model on the new data for a few epochs
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs, 
                    validation_data = (x_val, y_val), shuffle=True,
                    callbacks=callbacks)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

for layer in base_model.layers[:-5]:
    layer.trainable = False

for layer in base_model.layers[-5:]:
    layer.trainable = True
# compile the model (should be done *after* setting layers to non-trainable)
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
print("stage 2")
model.summary()
# train the model on the new data for a few epochs
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs, 
                    validation_data = (x_val, y_val), shuffle=True,
                    callbacks=callbacks)

for layer in base_model.layers[:-10]:
    layer.trainable = False

for layer in base_model.layers[-10:]:
    layer.trainable = True
# compile the model (should be done *after* setting layers to non-trainable)
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
print("stage 3")
model.summary()
# train the model on the new data for a few epochs
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs, 
                    validation_data = (x_val, y_val), shuffle=True,
                    callbacks=callbacks)