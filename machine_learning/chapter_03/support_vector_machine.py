from sklearn import datasets, metrics, svm

# 手書き数字データの読み込み
digits = datasets.load_digits()

# 3と8のデータ位置を求める
flag_3_8 = (digits.target == 3) + (digits.target == 8)

# 3と8のデータを取得
images = digits.images[flag_3_8]
labels = digits.target[flag_3_8]

# 3と8の画像データを1次元化
images = images.reshape(images.shape[0], -1)

# 分類器の生成
n_samples = len(flag_3_8[flag_3_8])
train_size = int(n_samples * 3 / 5)
classifier = svm.SVC(C=1.0, gamma=0.001)
classifier.fit(images[:train_size], labels[:train_size])

# 分類器の性能評価
expected = labels[train_size:]
predicted = classifier.predict(images[train_size:])

print('Accuracy:\n', metrics.accuracy_score(expected, predicted))
print('\nConfusion metrix:\n', metrics.confusion_matrix(expected, predicted))
print('\nPrecision:\n', metrics.precision_score(expected, predicted, pos_label=3))
print('\nRecall:\n', metrics.recall_score(expected, predicted, pos_label=3))
print('\nF-measure:\n', metrics.f1_score(expected, predicted, pos_label=3))