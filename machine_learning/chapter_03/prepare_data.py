import numpy as np
from sklearn import datasets

# 手書き数字データの読み込み
digits = datasets.load_digits()

# 3と8のデータ位置を求める
flag_3_8 = (digits.target == 3) + (digits.target == 8)

# 3と8のデータを取得
images = digits.images[flag_3_8]
labels = digits.target[flag_3_8]

# 3と8の画像データを1次元化
images = images.reshape(images.shape[0], -1)