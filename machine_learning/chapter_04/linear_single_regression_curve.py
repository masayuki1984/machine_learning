import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# 0~1までの乱数を100個作る
x = np.random.rand(100, 1)
# 値の範囲を -2~2 に変更
x = x * 4 - 2
# y=3x^2-2
y = 3 * x**2 - 2
# 標準正規分布(平均 0, 標準偏差 1)の乱数を加える
y += np.random.randn(100, 1)

model = linear_model.LinearRegression()
# x を二乗して渡す
model.fit(x**2, y)

plt.scatter(x, y, marker='+')
plt.scatter(x, model.predict(x**2), marker='o')

plt.show()

print(model.coef_)
print(model.intercept_)
print(model.score(x**2, y))