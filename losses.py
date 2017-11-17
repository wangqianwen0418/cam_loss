import keras.backend as K
lambda_center = 0.5
lambda_area = 0.5
def center_loss(y_true, y_pred):
    """
    # arguments
    y_true: centroids of bounding boxes, shape (batch_size, 2)
    y_pred: centroids of high attention area, shape (batch_size, 2)
    """ 
    x = y_true-y_pred
    x_abs = K.abs(x)
    x_bool = K.less_equal(x_abs, 1.0)
    smooth_l1 = x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5)
    # return K.sum(y_true * smooth_l1)/K.sum(y_true + K.epsilon())
    # bs = K.int_shape(y_true)[0]
    return K.sum(K.log(smooth_l1))

def area_loss(y_true, y_pred):
    """
    # arguments:
    y_true: bouding boxes (x,y,h,w), shape(batch_size, 4).
    y_pred: cam heatmap, shape (batch_size, 7, 7, 1).
    """
    y_pred = K.resize_images(
        y_pred,
        224/7,
        224/7,
        "channels_last"
    )
    y_pred = y_pred - K.min(y_pred)
    y_pred = y_pred/K.max()*255
    threshold = 150
    high_attention = K.greater_equal(y_pred, threshold)
    # p_in = 
    # p_out = 
    # p = p_out/p_in
    # return K.sum(K.log(p))
    x_l = K.argmin(high_attention, axis = 0)
    x_r = K.argmax(high_attention, axis = 0)
    x_t = K.argmin(high_attention, axis = 1)
    x_b = K.argmax(high_attention, axis = 1)

    if K.image_dim_ordering() == 'th':
        x = y_true[:, 4 * num_anchors:, :, :] - y_pred
        x_abs = K.abs(x)
        x_bool = K.less_equal(x_abs, 1.0)
        return lambda_rpn_regr * K.sum(
            y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
    else:
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])