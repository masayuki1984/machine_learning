import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# 0~1 までの乱数を100個作る
x = np.random.rand(100, 1)
# 値の範囲を -2~2 に変更
x = x * 2 - 1

y = 4 * x**3 - 3 * x**2 + 2 * x - 1
# 標準正規分布(平均 0, 標準偏差 1)の乱数を加える
y += np.random.randn(100, 1)

# 学習データ 30個
x_train = x[:30]
y_train = y[:30]

# テストデータ 70個
x_test = x[30:]
y_test = y[30:]

X_TRAIN = np.c_[x_train**9, x_train**8, x_train**7, x_train**6, x_train**5,
                x_train**4, x_train**3, x_train**2, x_train]

X_TEST = np.c_[x_test**9, x_test**8, x_test**7, x_test**6, x_test**5,
               x_test**4, x_test**3, x_test**2, x_test]

model = linear_model.LinearRegression()
model.fit(X_TRAIN, y_train)

plt.subplot(2, 2, 1)
plt.scatter(x, y, marker='+')
plt.title('all')

plt.subplot(2, 2, 2)
plt.scatter(x_train, y_train, marker='+')
plt.title('train')

plt.subplot(2, 2, 3)
plt.scatter(x_test, y_test, marker='+')
plt.title('test')

plt.tight_layout()
plt.show()

print(model.coef_)
print(model.intercept_)
print("X_TRAIN's score: ", model.score(X_TRAIN, y_train))

plt.scatter(x_train, y_train, marker='+')
plt.scatter(x_train, model.predict(X_TRAIN))
plt.show()

print("X_TEST's score: ", model.score(X_TEST, y_test))
plt.scatter(x_test, y_test, marker='+')
plt.scatter(x_test, model.predict(X_TEST))
plt.show()
