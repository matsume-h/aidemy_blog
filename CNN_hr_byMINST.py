#os のインポート
import os
#numpy,PIL,sklearn_util,datasets,tensorflowのインポート
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#別途に用意したCNNのクラス
import CNNclasses as cnn

#MNISTデータを取得、目的変数Xと説明変数yに分ける
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_X, mnist_y = mnist.train.images, mnist.train.labels
mnist_X = mnist_X.reshape((mnist_X.shape[0], 28, 28, 1))
#教師データとテストデータに分ける
train_X, valid_X, train_y, valid_y = train_test_split(mnist_X, mnist_y, test_size=0.1, random_state=42)

#層を用意
layers = [
    cnn.Conv((5, 5, 1, 20), tf.nn.relu),
    cnn.Pooling((1, 2, 2, 1)),
    cnn.Conv((5, 5, 20, 50), tf.nn.relu),
    cnn.Pooling((1, 2, 2, 1)),
    cnn.Flatten(),
    cnn.Dense(4*4*50, 10, tf.nn.softmax)
]

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
t = tf.placeholder(tf.float32, [None, 10])
#順伝播、出力、誤差関数、勾配降下法の定義
def f_props(layers, x):
    for layer in layers:
        x = layer.f_prop(x)
    return x

y = f_props(layers, x)

cost = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1)) # tf.log(0)によるnanを防ぐ
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

valid = tf.argmax(y, 1)
#訓練回数、訓練サイズ
n_epochs = 10
batch_size = 100
n_batches = train_X.shape[0]//batch_size

#訓練前にグラフの初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#訓練開始
for epoch in range(n_epochs):
    train_X, train_y = shuffle(train_X, train_y, random_state=42)
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})
    #訓練過程を出力
    pred_y_train = sess.run(valid, feed_dict={x: train_X, t: train_y})
    pred_y_valid = sess.run(valid, feed_dict={x: valid_X, t: valid_y})
    f1_train = f1_score(np.argmax(train_y, 1).astype('int32'), pred_y_train, average='macro')
    f1_valid = f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y_valid, average='macro')
    print('EPOCH:: %i, F1値(train): %.3f, F1値(valid): %.3f' % (epoch + 1, f1_train, f1_valid))1).astype('int32'), pred_y, average='macro')))

#画像の入っているフォルダを指定し、中身のファイル名を取得
filenames = sorted(os.listdir('handwrite_numbers'))
#フォルダ内の全画像をデータ化
img_test = np.empty((0, 28**2))
for filename in filenames:
    #画像ファイルを取得、グレースケールにしてサイズ変更
    img = Image.open('handwrite_numbers/' + filename).convert('L')
    resize_img = img.resize((84, 84))
    img_data256 = np.array([])
    #MNISTと同じサイズにする
    for _y in range(28):
        for _x in range(28):
            #1画素に縮小される範囲の明るさの二乗平均をとり、白黒反転
            crop = np.asarray(resize_img.crop(
                (_x * 3, _y * 3, _x * 3 + 3, _y * 3 + 3)))
            bright = 255 - int(crop.mean()**2 / 255)
            img_data256 = np.append(img_data256, bright)
    #画像データ内の最小値が0、最大値が1になるように計算
    min_bright = img_data256.min()
    max_bright = img_data256.max()
    img_data1 = (img_data256 - min_bright) / (max_bright - min_bright)
    img_test = np.r_[img_test, img_data1.astype(np.float32).reshape(1, -1)]

img_test = img_test.reshape((img_test.shape[0], 28, 28, 1))
#テストデータを実際に識別
pred_y = sess.run(valid, feed_dict={x: img_test})

true_X = []
for filename in filenames:
    true_X = true_X + [int(filename[:1])]
true_X = np.array(true_X)

score = np.ones(pred_y.size)[pred_y == true_X].sum() / pred_y.size

print('判別結果')
print('観測:', true_X)
print('予測:', pred_y)
print('正答率:', score)
