#osのインポート
import os
#numpy,PIL,sklearnのインポート
import numpy as np
from PIL import Image
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

#sklearnのデータセットからdigitsデータを取得、目的変数Xと説明変数yに分ける
digits = load_digits()
X = digits.data
y = digits.target
#教師データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)
#線形回帰のモデルを作る。教師データを使って学習
logreg = LogisticRegression()
logreg_model = logreg.fit(X_train, y_train)
#確認のため訓練結果を表示
print('訓練データのスコア:', logreg_model.score(X_train, y_train))
print('テストデータのスコア:', logreg_model.score(X_test, y_test))

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
    #画像データ内の最小値が0、最大値が16になるように計算
    min_bright = img_data256.min()
    max_bright = img_data256.max()
    img_data16 = (img_data256 - min_bright) / (max_bright - min_bright) * 16
    #加工した画像データの配列をまとめる
    img_test = np.r_[img_test, img_data16.astype(np.uint8).reshape(1, -1)]

X_true = []
for filename in filenames:
    X_true = X_true + [int(filename[:1])]
X_true = np.array(X_true)
#テストデータを識別
pred_logreg = logreg_model.predict(img_test)

print('判別結果')
print('観測:', X_true)
print('予測:', pred_logreg)
print('正答率:', logreg_model.score(img_test, X_true))
