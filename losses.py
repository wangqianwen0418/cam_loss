import keras.backend as K
import numpy as np

lambda_center = 0.5
lambda_area = 0.5


def my_loss(y_true, y_pred): 
    """
        # arguments:
        y_true: bouding boxes (x,y,h,w), shape(batch_size, 4).
        y_pred: cam heatmap, shape (batch_size, 7, 7, 1).
        """  
    batch_size = K.int_shape(y_pred)[0]
    pos_x = np.zeros((batch_size, 7,7, 1))
    pos_y =np.zeros((batch_size, 7,7, 1))
    for i in range(7):
        pos_x[:,i,:] = i
        pos_y[:,:,i] = i
    pos_x = K.variable(value=pos_x)
    pos_y = K.variable(value=pos_y)

    def center_loss(y_true, y_pred):
        
        bbox_c = y_true[:, :2]
        sum_v = K.expand_dims(K.sum(y_pred, axis=(1,2,3)))
        x_c = K.expand_dims(K.sum(y_pred * pos_x, axis=(1,2,3)))/sum_v
        y_c = K.expand_dims(K.sum(y_pred * pos_y, axis=(1,2,3)))/sum_v
        # print(K.int_shape(K.concatenate([x_c, y_c], axis=1)))
        heatmap_c = K.concatenate([x_c, y_c], axis=1) #shape(bs, 2)
        x = heatmap_c - bbox_c
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), K.floatx())
        smooth_l1 = x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5)
        # return K.sum(y_true * smooth_l1)/K.sum(y_true + K.epsilon())
        # bs = K.int_shape(y_true)[0]
        return K.sum(K.log(smooth_l1), axis=1)

    def area_loss(y_true, y_pred):
        h0 =224
        w0=224
        y_pred = K.resize_images(
            y_pred,
            h0/7,
            w0/7,
            "channels_last"
        )
        y_pred = y_pred[:,:,:, 0]
        # normalize heatmap
        min = K.expand_dims(K.expand_dims(K.min(y_pred, axis=(1,2))))
        min = K.repeat_elements(min, rep=h0, axis=1)
        min = K.repeat_elements(min, rep=w0, axis= 2)
        max = K.expand_dims(K.expand_dims(K.max(y_pred, axis=(1,2))))
        max = K.repeat_elements(max, rep=h0, axis=1)
        max = K.repeat_elements(max, rep=w0, axis= 2)
        y_pred = y_pred - min
        y_pred = y_pred/max
        
        y_box = np.zeros(K.int_shape(y_pred))
        for i in range(batch_size):
            box_info = K.eval(y_true[i,:]) # (x, y, w,h)
            [x, y, w, h] = box_info
            # print("box", K.eval(x-w/2), K.eval(x+w/2),K.eval(y-h/2),K.eval(y+h/2))
            y_box[i, int(x-w/2):int(x+w/2), int(y-h/2):int(y+h/2)] = 1
        y_box = K.variable(value=y_box)

        v_in = K.sum(y_pred * y_box, axis=(1,2))
        v_out = K.sum(y_pred, axis=(1,2)) - v_in
        return K.log(v_out/v_in)

        # threshold = 150
        # high_attention = K.greater_equal(y_pred, threshold)
        # # p_in = 
        # # p_out = 
        # # p = p_out/p_in
        # # return K.sum(K.log(p))
        # x_l = K.argmin(high_attention, axis = 0)
        # x_r = K.argmax(high_attention, axis = 0)
        # x_t = K.argmin(high_attention, axis = 1)
        # x_b = K.argmax(high_attention, axis = 1)

        # if K.image_dim_ordering() == 'th':
        #     x = y_true[:, 4 * num_anchors:, :, :] - y_pred
        #     x_abs = K.abs(x)
        #     x_bool = K.less_equal(x_abs, 1.0)
        #     return lambda_rpn_regr * K.sum(
        #         y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
        # else:
        #     x = y_true[:, :, :, 4 * num_anchors:] - y_pred
        #     x_abs = K.abs(x)
        #     x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        #     return lambda_rpn_regr * K.sum(
        #         y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])
    # def total_loss(y_true, y_pred):
    #     print("shape of area loss", K.int_shape(area_loss(y_true, y_pred)))
    #     print("shape of center loss", K.int_shape(center_loss(y_true, y_pred)))
    #     return lambda_area * area_loss(y_true, y_pred) + lambda_center * center_loss(y_true, y_pred)
    total_loss = lambda_area * area_loss(y_true, y_pred) + lambda_center * center_loss(y_true, y_pred)
    return total_loss

