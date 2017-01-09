import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# 0~1までの乱数を100個作る
x1 = np.random.rand(100, 1)
# 値の範囲を -2~2 に変更
x1 = x1 * 4 - 2
# x2についても同様に行う
x2 = np.random.rand(100, 1)
x2 = x2 * 4 - 2

y = 3 * x1 - 2 * x2 + 1

# 標準正規分布(平均 0, 標準偏差 1)の乱数を加える
y += np.random.randn(100, 1)

# [[x1_1, x2_1], [x1_2, x2_2], ..., [x1_100, x2_100]] という形に変換
x1_x2 = np.c_[x1, x2]

model = linear_model.LinearRegression()
model.fit(x1_x2, y)

# 求めた回帰式で予測
y_ = model.predict(x1_x2)


plt.subplot(1, 2, 1)
plt.scatter(x1, y, marker='+')
plt.scatter(x1, y_, marker='o')
plt.xlabel('x1')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.scatter(x2, y, marker='+')
plt.scatter(x2, y_, marker='o')
plt.xlabel('x2')
plt.ylabel('y')

plt.tight_layout()
plt.show()

print(model.coef_)
print(model.intercept_)
print(model.score(x1_x2, y))
