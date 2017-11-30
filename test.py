from model import cam_model
from keras import optimizers 
from losses import my_loss
batch_size = 32

model = cam_model()
model.summary()
# print(K.int_shape(y))

model.compile(optimizer=optimizers.Adam(),
              loss={"label":"categorical_crossentropy", "map":my_loss(batch_size)},
              metrics=['accuracy', 'accuracy'])


# fc_weights = model.get_layer(name="fc1000").get_weights()
# print(fc_weights[0].shape)# shape[2048, 10000]
# # fc_copy = 