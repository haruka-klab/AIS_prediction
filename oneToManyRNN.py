import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np
import pandas as pd
import dataConverter
import datetime

# model定義
class oneToManyRNN(tf.keras.Model):
    def __init__(self,input_shape):
        super().__init__()
        output_dim = input_shape[1]
        self.l1 = kl.Dense(output_dim,input_shape=input_shape)
        self.l2 = kl.Dense(output_dim)
        self.l3 = kl.Dense(output_dim)

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
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss = tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()])

    def train(self, xtrain, ytrain, batch_size, epochs):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S" +"dddk")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        his = self.model.fit(xtrain,ytrain,batch_size=batch_size, epochs=epochs,callbacks=[tensorboard_callback])
        his = pd.DataFrame(his.history)
        #showHistory(his)

    def predict(self,x):
        return self.model.predict(x)

# ヒストリーの表示
def showHistory(his):
    his.plot()

# https://qiita.com/hima_zin331/items/2adba781bc4afaae5880
# sampleソースコード
# メイン処理
if __name__ == "__main__":

    # データ準備
    tf.random.set_seed(345)
    np.random.seed(345)

    n = 200
    x = np.random.random((n,6))
    y = np.concatenate([x*2,x*3,x*4],1)

    # 実際のデータ
    # dac = dataConverter("file pass")
    # x,y = dac.convert()

    # Trainer (model) のインスタンス化
    # 学習、実行、保存
    tr = trainer((None,6))
    tr.train(x,y,batch_size=50,epochs=2000)

    print(tr.predict([[1,2,3,4,5,6]]))
