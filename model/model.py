import keras
from keras.engine.topology import Layer
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras import layers
from keras.models import Model
from keras.layers import dot, Reshape
from keras import optimizers 
import keras.preprocessing.image.ImageDataGenerator as ImageDataGenerator

import keras.backend as K
# K.set_learning_phase(1)  # 1 for training, 0 for testing

from losses import area_loss
from get_coco import COCOGenerator, COCODataset

import os


batch_size = 32
target_size = (224, 224)
COI = ['cat']
epochs = 50
iter_epo = 3
rel_layers = [5, 5, 5]
num_classes = len(COI) + 1


class FC_layer(Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        super(FC_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel= self.add_weight(shape=(input_dim, self.units),
                                      initializer="glorot_uniform",
                                      name='kernel')
        super(FC_layer, self).build(input_shape)

    def call(self, x):
        return K.dot(x, self.kernel)
    #     if K.ndim(x)==2:
    #         return K.dot(x, self.kernel)
    #     elif K.ndim(x)==4:
    #         (bs, h, w, nc) = list(K.int_shape(x))
    #         reshape_x = K.reshape(x, (None, h*w, nc))
    #         y = K.dot(reshape_x, self.kernel)
    #         return K.reshape(y, (bs, h, w, self.units))
    #     else:
    #         raise TypeError("wrong input shape") 

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

def get_map(inputs):
    (bs, h, w, num_c) = K.int_shape(inputs[0])
    maps = Reshape((num_c, h*w))(inputs[0]) # shape (None, 1000, 49)
    labels = inputs[1]
    print("labels shape", K.int_shape(labels))
    idx = K.expand_dims(K.one_hot(K.argmax(labels, axis=1), K.int_shape(labels)[1]))# shape (None, 1000)
    map = dot([idx, maps], axes=1) # get heap map for the top1 prediction, shape (None, 49)
    map = Reshape((h, w, 1))(map) # shape, (None, 7, 7, 1)
    # print("map shape", K.int_shape(map))
    return map


# base_model = ResNet50(weights="imagenet", input_shape=target_size + (3,), include_top=False,pooling="avg")
# last_conv = "activation_49" # for resnet50
base_model = VGG19(weights="imagenet", input_shape=target_size + (3,), include_top=False,pooling="avg")
last_conv = "block5_pool" # for vgg19 
conv_features = base_model.get_layer(name=last_conv).output # get the output of last conv, shape (batch_size, 7, 7, 2048)

x = base_model.output # the results after global avg, shape(batch_size, 2048)
fc_layer = FC_layer(num_classes, name="last_fc") # custom fully connected layer, it is shared between "x" and "conv_feature"
x = fc_layer(x)
labels = layers.Activation("softmax", name="label")(x)

maps = fc_layer(conv_features) # shape(7,7,1000)
heat_map = layers.Lambda(get_map, name="map")([maps, x])
model = Model(base_model.input, [labels, heat_map])
model.summary()

model.compile(optimizer=optimizers.Adam(),
              loss={"label":"categorical_crossentropy", "map":area_loss},
              metrics=['accuracy', 'accuracy'])

# get data

coco = COCODataset(data_dir="../data/coco/", COI=COI, img_set="train")
coco_generator = COCOGenerator(coco, batch_size, target_size)

model.fit_generator(coco_generator.generate(),
            samples_per_epoch=samples_per_epoch,
            nb_epoch=nb_epoch)
