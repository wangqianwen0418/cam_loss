import keras.backend as K
import numpy as np

lambda_center = 0.5
lambda_area = 0.5


def my_loss(batch_size): 
    """
    # arguments:
    y_true: bouding boxes (x,y,h,w), shape(batch_size, h0, w0,1).
    y_pred: cam heatmap, shape (batch_size, 7, 7, 1).
    """  

    # def center_loss(y_true, y_pred):
    #     return K.sum(y_pred)
    #     (_, w, h, num_c) = K.int_shape(y_pred)
    #     pos_x = np.zeros(K.int_shape(y_pred))
    #     pos_y =np.zeros(K.int_shape(y_pred))
    #     for i in range(w):
    #         pos_x[:,i,:] = i
    #     for i in range(h):
    #         pos_y[:,:,i] = i
    #     pos_x = K.variable(value=pos_x)
    #     pos_y = K.variable(value=pos_y)
        
    #     bbox_c = y_true[:, :2]
    #     sum_v = K.expand_dims(K.sum(y_pred, axis=(1,2,3)))
    #     x_c = K.expand_dims(K.sum(y_pred * pos_x, axis=(1,2,3)))/sum_v
    #     y_c = K.expand_dims(K.sum(y_pred * pos_y, axis=(1,2,3)))/sum_v
    #     # print(K.int_shape(K.concatenate([x_c, y_c], axis=1)))
    #     heatmap_c = K.concatenate([x_c, y_c], axis=1) #shape(bs, 2)
    #     x = heatmap_c - bbox_c
    #     x_abs = K.abs(x)
    #     x_bool = K.cast(K.less_equal(x_abs, 1.0), K.floatx())
    #     smooth_l1 = x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5)
    #     # return K.sum(y_true * smooth_l1)/K.sum(y_true + K.epsilon())
    #     # bs = K.int_shape(y_true)[0]
    #     return K.sum(K.log(smooth_l1), axis=1)

    def area_loss(y_true, y_pred):
        (_, h, w, num_c) = K.int_shape(y_pred)
        # (_, w0, h0, num_c) = K.int_shape(y_true)
        h0 = 224
        w0 = 224
        y_pred = K.resize_images(
            y_pred,
            h0/h,
            w0/w,
            "channels_last"
        )
        # normalize heatmap
        # min = K.expand_dims(K.expand_dims(K.min(y_pred, axis=(1,2))))
        # min = K.repeat_elements(min, rep=h0, axis=1)
        # min = K.repeat_elements(min, rep=w0, axis= 2)
        # max = K.expand_dims(K.expand_dims(K.max(y_pred, axis=(1,2))))
        # max = K.repeat_elements(max, rep=h0, axis=1)
        # max = K.repeat_elements(max, rep=w0, axis= 2)

        min = K.min(y_pred)
        max = K.max(y_pred)
        y_pred = y_pred - min
        y_pred = y_pred/max #shape (None, h0,w0)

        # y_box = np.zeros((batch_size, h, w, num_c))
        # for i in range(batch_size):
        #     box_info = K.eval(y_true[i,:]) # (x, y, w,h)
        #     [x, y, w, h] = box_info
        #     # print("box", K.eval(x-w/2), K.eval(x+w/2),K.eval(y-h/2),K.eval(y+h/2))
        #     y_box[i, int(x-w/2):int(x+w/2), int(y-h/2):int(y+h/2)] = 1
        # y_box = K.variable(value=y_box)

        v_in = K.sum(y_pred*y_true, axis=(1,2,3))
        v_out = K.sum(y_pred, axis=(1,2,3)) - v_in
        return K.log(v_out/v_in)
    
    def total_loss(y_true, y_pred):
        return lambda_area * area_loss(y_true, y_pred) + lambda_center * center_loss(y_true, y_pred)
    # total_loss = lambda_area * area_loss(y_true, y_pred) + lambda_center * center_loss(y_true, y_pred)
    return total_loss


def area_loss(y_true, y_pred):
        (_, h, w, num_c) = K.int_shape(y_pred)
        # (_, w0, h0, num_c) = K.int_shape(y_true)
        h0 = 224
        w0 = 224
        y_pred = K.resize_images(
            y_pred,
            h0/h,
            w0/w,
            "channels_last"
        )
        # normalize heatmap
        # min = K.expand_dims(K.expand_dims(K.min(y_pred, axis=(1,2))))
        # min = K.repeat_elements(min, rep=h0, axis=1)
        # min = K.repeat_elements(min, rep=w0, axis= 2)
        # max = K.expand_dims(K.expand_dims(K.max(y_pred, axis=(1,2))))
        # max = K.repeat_elements(max, rep=h0, axis=1)
        # max = K.repeat_elements(max, rep=w0, axis= 2)

        min = K.min(y_pred)
        max = K.max(y_pred)
        y_pred = y_pred - min
        y_pred = y_pred/max #shape (None, h0,w0)

        # y_box = np.zeros((batch_size, h, w, num_c))
        # for i in range(batch_size):
        #     box_info = K.eval(y_true[i,:]) # (x, y, w,h)
        #     [x, y, w, h] = box_info
        #     # print("box", K.eval(x-w/2), K.eval(x+w/2),K.eval(y-h/2),K.eval(y+h/2))
        #     y_box[i, int(x-w/2):int(x+w/2), int(y-h/2):int(y+h/2)] = 1
        # y_box = K.variable(value=y_box)

        v_in = K.sum(y_pred*y_true, axis=(1,2,3))
        v_out = K.sum(y_pred, axis=(1,2,3)) - v_in
        return K.log(v_out/v_in)