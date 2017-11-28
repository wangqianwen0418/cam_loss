import keras
from keras.engine.topology import Layer
from keras.applications.resnet50 import ResNet50
from keras import layers
from keras.models import Model

import keras.backend as K

from losses import my_loss

num_classes = 1000

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
    maps = keras.layers.Reshape((1000, 7, 7))(inputs[0])
    labels = inputs[1]
    if K.ndim(maps)!=4:
        raise TypeError("maps is {}, should have 4 dimesions".format(K.int_shape(maps)))
    idx = K.one_hot(K.argmax(labels, axis=1), K.int_shape(labels)[-1])
    map = K.batch_dot(idx, maps, axes=[1,1]) # get heap map for the top1 prediction
    print("map shape", K.int_shape(maps))
    # print("labels shape", K.int_shape(labels))
    print("idx", K.int_shape(idx))
    # print("idx",K.eval(idx))
    # map = maps[:,:,:,idx]
    return maps


base_model = ResNet50(weights=None, input_shape=(224,224,3), include_top=False,pooling="avg")
last_conv = "activation_49"
conv_features = base_model.get_layer(name=last_conv).output # get the output of last conv, shape (batch_size, 7, 7, 2048)

x = base_model.output # the results after global avg, shape(batch_size, 2048)
fc_layer = FC_layer(num_classes, name="last_fc") # custom fully connected layer, it is shared between "x" and "conv_feature"
x = fc_layer(x)
labels = layers.Activation("softmax")(x)

maps = fc_layer(conv_features) # shape(7,7,1000)
heat_map = layers.Lambda(get_map)([maps, x])

model = Model(base_model.input, [labels, heat_map])
# model.summary()
# print(K.int_shape(y))

# fc_weights = model.get_layer(name="fc1000").get_weights()
# print(fc_weights[0].shape)# shape[2048, 10000]
# # fc_copy = 