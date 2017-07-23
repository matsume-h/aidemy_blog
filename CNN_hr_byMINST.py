#os,numpy,PILのインポート
import os
from PIL import Image

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

import CNNclasses as cnn

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_X, mnist_y = mnist.train.images, mnist.train.labels
mnist_X = mnist_X.reshape((mnist_X.shape[0], 28, 28, 1))

train_X, valid_X, train_y, valid_y = train_test_split(mnist_X, mnist_y, test_size=0.1, random_state=42)


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

def f_props(layers, x):
    for layer in layers:
        x = layer.f_prop(x)
    return x

y = f_props(layers, x)

cost = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1)) # tf.log(0)によるnanを防ぐ
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

valid = tf.argmax(y, 1)

n_epochs = 10
batch_size = 100
n_batches = train_X.shape[0]//batch_size

init = tf.global_variables_initializer()


sess = tf.Session()

sess.run(init)
for epoch in range(n_epochs):
    train_X, train_y = shuffle(train_X, train_y, random_state=42)
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})
    pred_y, valid_cost = sess.run([valid, cost], feed_dict={x: valid_X, t: valid_y})
    print('EPOCH:: %i, 誤差: %.3f, F1値: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y, average='macro')))



foldername = 'handwrite_numbers'

filenames = sorted(os.listdir(foldername))
#フォルダ内の全画像をデータ化
img_test = np.empty((0, 28**2))
for filename in filenames:

    img = Image.open(foldername + '/' + filename).convert('L')
    resize_img = img.resize((84, 84))
    img_data256 = np.array([])
    for _y in range(28):
        for _x in range(28):
            crop = np.asarray(resize_img.crop(
                (_x * 3, _y * 3, _x * 3 + 3, _y * 3 + 3)))
            bright = 255 - np.min(crop)
            img_data256 = np.append(img_data256, bright)

    #画像データ内の最小値が0、最大値が1になるように計算
    min_bright = img_data256.min()
    max_bright = img_data256.max()
    img_data1 = (img_data256 - min_bright) / (max_bright - min_bright)
    img_test = np.r_[img_test, img_data1.astype(np.float32).reshape(1, -1)]

img_test = img_test.reshape((img_test.shape[0], 28, 28, 1))

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
