#os のインポート
import os
#numpy,PIL,datasets,sklearn_util,tensorflowのインポート
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import tensorflow as tf
#別途に用意したCNNのクラス
import CNNclasses as cnn

#sklearnのデータセットからdigitsデータを取得、目的変数Xと説明変数yに分ける
digits = load_digits()
digit_X = digits.data.reshape(
    (digits.data.astype(np.float32).shape[0], 8, 8, 1)) / digits.data.max()
digit_y = np.zeros((digits.target.size, 10))
index = np.arange(digit_y.shape[0])
digit_y[index, digits.target[index]] = 1
#教師データとテストデータに分ける
train_X, valid_X, train_y, valid_y = train_test_split(
    digit_X, digit_y, test_size=0.1, random_state=0)

#層を用意 畳込み->Pooling->平滑化->全結合
layers = [
    cnn.Conv((5, 5, 1, 50), tf.nn.relu),
    cnn.Pooling((1, 2, 2, 1)),
    cnn.Flatten(),
    cnn.Dense(2 * 2 * 50, 10, tf.nn.softmax)
]

x = tf.placeholder(tf.float32, [None, 8, 8, 1])
t = tf.placeholder(tf.float32, [None, 10])
#順伝播の定義
def f_props(layers, x):
    for layer in layers:
        x = layer.f_prop(x)
    return x
#出力
y = f_props(layers, x)
#誤差関数
cost = -tf.reduce_mean(tf.reduce_sum(t *
                                     tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1))
#誤差を最小化するように訓練するよう設定
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
#出力を1-of-k方式から10進の配列へ
valid = tf.argmax(y, 1)
#訓練回数、訓練サイズ
n_epochs = 300
batch_size = 100
n_batches = train_X.shape[0] // batch_size

#訓練前にグラフの初期化
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#訓練開始
for epoch in range(n_epochs):
    train_X, train_y = shuffle(train_X, train_y, random_state=42)
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        #出力結果の誤差を最小化するようにバイアスを調整
        sess.run(train, feed_dict={
                 x: train_X[start:end], t: train_y[start:end]})
    #訓練過程を出力
    if epoch % 100 == 99:
        pred_y_train = sess.run(valid, feed_dict={x: train_X, t: train_y})
        pred_y_valid = sess.run(valid, feed_dict={x: valid_X, t: valid_y})
        f1_train = f1_score(np.argmax(train_y, 1).astype(
            'int32'), pred_y_train, average='macro')
        f1_valid = f1_score(np.argmax(valid_y, 1).astype(
            'int32'), pred_y_valid, average='macro')
        print('EPOCH:: %i, F1値(train): %.3f, F1値(valid): %.3f' % (epoch + 1, f1_train, f1_valid))1).astype('int32'), pred_y, average='macro')))

#画像の入っているフォルダを指定し、中身のファイル名を取得
filenames = sorted(os.listdir('handwrite_numbers'))
#フォルダ内の全画像をデータ化
img_test = np.empty((0, 64))
for filename in filenames:
    #画像ファイルを取得、グレースケールにして白黒反転し、余白を削る
    img = Image.open('handwrite_numbers/' + filename).convert('L')
    img = img.point(lambda x: 255 if x < 200 else 0)
    #文字の横幅、縦幅をとる
    width = img.getbbox()[2] - img.getbbox()[0]
    height = img.getbbox()[3] - img.getbbox()[1]
    #文字の余白をトリミング
    croped_img = img.crop(img.getbbox())
    #文字幅にあったサイズの正方形を用意、中心にトリミングした文字を貼り付け
    fieldsize = max(width, height)
    imgfield = Image.new('L', (fieldsize, fieldsize))
    imgfield.paste(croped_img, ((fieldsize - width) // 2, (fieldsize - height) // 2))
    #サイズを変更
    resize_img = imgfield.resize((64, 64))
    img_data256 = np.array([])
    #サイズを更に縮めて配列を作り、sklearnのdigitsと同じ型にする
    #64画素の1画素ずつ明るさをプロット
    for _y in range(8):
        for _x in range(8):
            #1画素に縮小される範囲の明るさの平均をとる
            crop = np.asarray(resize_img.crop(
                (_x * 8, _y * 8, _x * 8 + 8, _y * 8 + 8)))
            bright = int(crop.mean())
            img_data256 = np.append(img_data256, bright)
    # 画像データ内の最小値が0、最大値が1になるように計算
    min_bright = img_data256.min()
    max_bright = img_data256.max()
    img_data16 = (img_data256 - min_bright) / (max_bright - min_bright)
    # 加工した画像データの配列をまとめる
    img_test = np.r_[img_test, img_data16.astype(np.float32).reshape(1, -1)]

img_test = img_test.reshape((img_test.shape[0], 8, 8, 1))
#テストデータを識別
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
