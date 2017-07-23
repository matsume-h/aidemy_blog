import numpy as np
import tensorflow as tf
#CNNの層クラスを定義
#畳み込み層
class Conv:
    def __init__(self, filter_shape, function=lambda x: x, strides=[1, 1, 1, 1], padding='VALID'):
        # Xavier のバイアス初期化
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.W = tf.Variable(np.random.uniform(
            low=-np.sqrt(6 / (fan_in + fan_out)),
            high=np.sqrt(6 / (fan_in + fan_out)),
            size=filter_shape
        ).astype('float32'), name='W')
        self.b = tf.Variable(
            np.zeros((filter_shape[3]), dtype='float32'), name='b')
        self.function = function
        self.strides = strides
        self.padding = padding

    def f_prop(self, x):
        u = tf.nn.conv2d(x, self.W, strides=self.strides,
                         padding=self.padding) + self.b
        return self.function(u)
#Pooling層
class Pooling:
    def __init__(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def f_prop(self, x):
        return tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)
#平滑化層
class Flatten:
    def f_prop(self, x):
        return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
#全結合層
class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        # Xavier Initialization
        self.W = tf.Variable(np.random.uniform(
            low=-np.sqrt(6 / (in_dim + out_dim)),
            high=np.sqrt(6 / (in_dim + out_dim)),
            size=(in_dim, out_dim)
        ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function

    def f_prop(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)
