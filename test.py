import numpy as np 
from losses import my_loss
import keras.backend as K
batch_size = 32
box = np.random.randint(5, size=(batch_size, 4))
box = K.variable(value=box)
pred = np.random.randint(10, size=(batch_size, 7, 7, 1))
pred = K.variable(value=pred)

loss = my_loss(batch_size)(box, pred)
print(K.int_shape(loss))
