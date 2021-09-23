import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np
import pandas as pd
import dataConverter

# model定義
class oneToManyRNN(tf.keras.Model):
    def __init__(self,input_shape):
        super().__init__()
        self.input_shape=input_shape
        self.output_shape = input_shape
        self.l1 = kl.Dense(self.output_shape,input_shape=input_shape,activation="tanh")
        self.l2 = kl.Dense(self.output_shape,input_shape=input_shape,activation="tanh")
        self.l3 = kl.Dense(self.output_shape,input_shape=input_shape,activation="tanh")

    def call(self,x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        return tf.concat([h1,h2,h3],1)

# modelを作ってトレーニング
class trainer(object):
    def __init__(self,input_shape):
        self.model = oneToManyRNN(input_shape)
        self.model.build(input_shape=(input_shape))
        self.model.compile(optimizer=tf.keras.optimizers.SGD(),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.mean_squared_error()])

    def train(self, xtrain, ytrain, batch_size, epochs):
        his = self.model.train(xtrain,ytrain,batch_size=batch_size, epochs=epochs)

# ヒストリーの表示


# https://qiita.com/hima_zin331/items/2adba781bc4afaae5880
# sampleソースコード
# メイン処理
if __name__ == "__main__":

    # データ準備
    tf.random.set_seed(345)
    np.random.seed(345)

    n = 200
    x = np.random.random((n,6))
    y = x

    # 実際のデータ
    # dac = dataConverter("file pass")
    # x,y = dac.convert()

    # Trainer (model) のインスタンス化
    # 学習、実行、保存
    tr = trainer((None,6))
    tr.train(x,y,batch_size=3,epochs=10)
