"""
    カスタマイズ層たち
    作り方
    tf.keras.Layers.Layerを継承し、
    __init__、build、call、get_configを定義する
"""
import tensorflow as tf
import numpy as np

# 雛形
class hina(tf.keras.layers.Layer):
    def __init__(self, output_dim, custum_params=0.1, **kwargs):
        self.output_dim = output_dim
        # 処理に必要な特別な変数を用意する場合
        self.custum_params = custum_params
        super(hina, self).__init__(**kwargs)

    def build(self, input_shape):
        #処理に必要な重みを定義する 典型的なのものはw と b
        self.w = self.add_weight(name = 'weights',
                                 shape = (input_shape[1], self.output_dim),
                                 initializer = 'random_normal',
                                 trainable = True)
        self.b = self.add_weight(shape = (self.output_dim,),
                                 initializer = 'zeros',
                                 trainable = True)
    
    def call(self, inputs, training=False):
        #計算グラフを定義する
        z = inputs
        return z
    
    def get_config(self):
        config = super(hina, self).get_config()
        config.update({'output_dim':self.output_dim,
                       'custum_params':self.custum_params})
        return config

# skip conection ただしデンスのものになっている
# ぶっちゃけモデルかすればよくね、と思っている
class skipConection(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(skipConection, self).__init__(**kwargs)

    def build(self, input_shape):
        #処理に必要な重みを定義する 典型的なのものはw と b
        self.w = self.add_weight(name = 'weights',
                                 shape = (input_shape[1], self.output_dim),
                                 initializer = 'random_normal',
                                 trainable = True)
        self.b = self.add_weight(shape = (self.output_dim,),
                                 initializer = 'zeros',
                                 trainable = True)
    
    def call(self, inputs, training=False):
        #計算グラフを定義する
        z = tf.matmul(inputs, self.w) + self.b + inputs
        return z
    
    def get_config(self):
        config = super(skipConection, self).get_config()
        config.update({'output_dim':self.output_dim})
        return config

"""
sample code (how to use)

model = tf.keras.Sequential([
    hina(3, custum_params= 0.01),
    tf.keras.layers.Dense(units=1, activation="relu")])

model.bulid(input_shape=(None,2))

model.compile(optimizer = keras.optimizers.SGD(),
            loss = tf.kers.losses.BinaryCrossentropy(),
            metrics = [tf.keras.metrics.BinaryAccuracy()])

訓練
保存 or 利用
"""