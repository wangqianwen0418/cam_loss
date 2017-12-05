import keras
from keras.engine.topology import Layer
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras import layers
from keras.models import Model
from keras.layers import dot, Reshape
from keras import optimizers 
from keras.preprocessing.image  import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import SGD

from keras import activations, initializers, regularizers, constraints
from keras.engine import InputSpec

import keras.backend as K
# K.set_learning_phase(1)  # 1 for training, 0 for testing

from losses import area_loss
from get_coco import COCOData

import os
import argparse

class FC_layer(Layer):
    def __init__(self, units, 
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs):
        super(FC_layer, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel= self.add_weight(shape=(input_dim, self.units),
                                      initializer="glorot_uniform",
                                      name='kernel')
        super(FC_layer, self).build(input_shape)

    def call(self, x):
        return K.dot(x, self.kernel)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(FC_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
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

# def get_map(inputs):
#     (bs, h, w, num_c) = K.int_shape(inputs[0])
#     maps = Reshape((num_c, h*w))(inputs[0]) # shape (None, num_classes, 49)
#     labels = inputs[1]
#     idx = K.expand_dims(K.one_hot(K.argmax(labels, axis=1), K.int_shape(labels)[1]))# shape (None, 1000)
#     map = dot([idx, maps], axes=1) # get heap map for the top1 prediction, shape (None, 49)
#     map = Reshape((h, w, 1))(map) # shape, (None, 7, 7, 1)
#     h0 = 224
#     w0 = 224
#     map = K.resize_images(
#         map,
#         target_size[0]/h,
#         target_size[1]/w,
#         "channels_last"
#     )
#     # print("map shape", K.int_shape(map))
#     return map
def resize(input):
    (bs, h, w, num_c) = K.int_shape(input)
    map = K.resize_images(
        input,
        224/7,
        224/7,
        "channels_last"
    )
    # print("map shape", K.int_shape(map))
    return map
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--bbox", type=lambda s: s.lower() in ['true', 't', 'yes', '1'], required=True)
    args = parser.parse_args()

    exp = args.exp
    bbox = args.bbox

    batch_size = 16
    target_size = (224, 224)
    COI = ['cat']
    epochs = 30
    r_blk = 1 # number of blocks to learn
    model_name = "vgg19"
    #number of layers at each block
    if model_name == "vgg19":
        blk_layers = [4,3,5,5,6]
    else:
        blk_layers =  [5, 12, 10, 10, 12, 10, 10, 10, 12,  10, 10, 10, 10, 10, 12, 10, 12]
    num_classes = 1

    if model_name == "vgg19":
        base_model = VGG19(weights="imagenet", input_shape=target_size + (3,), include_top=False,pooling="avg")
        last_conv = "block5_pool" # for vgg19 
    else:
        base_model = ResNet50(weights="imagenet", input_shape=target_size + (3,), include_top=False,pooling="avg")
        last_conv = "activation_49" # for resnet50
    conv_features = base_model.get_layer(name=last_conv).output # get the output of last conv, shape (batch_size, 7, 7, 2048)

    x = base_model.output # the results after global avg, shape(batch_size, 2048)
    # x = keras.layers.Dense(512, activation='relu')(x)
    # x = keras.layers.Dense(256, activation='relu')(x)
    fc_layer = FC_layer(num_classes, name="last_fc") # custom fully connected layer, it is shared between "x" and "conv_feature"
    x = fc_layer(x)
    labels = layers.Activation("sigmoid", name="label")(x)

    heat_map = fc_layer(conv_features) # shape(7,7,num_classes)
    # heat_map = layers.Activation('relu')(heat_map)
    heat_map = layers.Lambda(resize , name="map")(heat_map)
    # heat_map = layers.Lambda(get_map, name="map")([maps, x])

    for layer in base_model.layers:
        layer.trainable = False

    # get data
    cocodata_train = COCOData(data_dir="../data/coco/", COI=['cat'], img_set="train", batch_size=batch_size, target_size=target_size)
    cocodata_val = COCOData(data_dir="../data/coco/", COI=['cat'], img_set="val", batch_size=batch_size, target_size=target_size)

    ## callbacks to save the best model
    if bbox:
        monitor = "val_label_acc"
    else:
        monitor = "val_acc"
    check_point = ModelCheckpoint("save_model/{}_{}.h5".format(model_name, exp), monitor=monitor, verbose=0, save_best_only=True, save_weights_only=False)
    tf_log = TensorBoard(log_dir='save_model/tf_logs_{}_{}'.format(model_name, exp), batch_size=batch_size, write_graph=True)
    callbacks = [check_point, tf_log]


    if bbox:
        model = Model(base_model.input, [labels, heat_map])

        model.compile(optimizer=optimizers.Adam(),
                    loss={"label":"binary_crossentropy", "map":area_loss},
                    loss_weights = [1, 0.2],
                    metrics=['accuracy'])

    else:
        model = Model(base_model.input, labels)
        model.compile(optimizer='adam',
                    loss="binary_crossentropy",
                    metrics=['accuracy'])

        

    model.fit_generator(cocodata_train.generate(bbox),
                steps_per_epoch=cocodata_train.num_images//batch_size,
                validation_data = cocodata_val.generate(bbox),
                validation_steps = cocodata_val.num_images//batch_size,
                epochs=epochs, callbacks=callbacks)


    for i in range(r_blk):
        i = i+1
        if i==r_blk:
            cbs = callbacks
        else:
            cbs = None
        num_free_layers = sum(blk_layers[-i:])
        for layer in base_model.layers[:-num_free_layers]:
            layer.trainable = False
        for layer in base_model.layers[-num_free_layers:]:
            layer.trainable = True
        if bbox:
            model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
                    loss={"label":"binary_crossentropy", "map":area_loss},
                    loss_weights = [1, 0.4],
                    metrics=['accuracy'])
        else:
            model.compile(optimizer=SGD(lr=1e-5, momentum=0.9),
                    loss="binary_crossentropy",
                    metrics=['accuracy'])
        print("free the last {} layers".format(num_free_layers))
        model.fit_generator(cocodata_train.generate(bbox),
                steps_per_epoch=cocodata_train.num_images//batch_size,
                validation_data = cocodata_val.generate(bbox),
                validation_steps = cocodata_val.num_images//batch_size,
                epochs=epochs, callbacks=cbs)


