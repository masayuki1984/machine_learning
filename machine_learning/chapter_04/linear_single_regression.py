import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# 0~1までの乱数を100個作る
x = np.random.rand(100, 1)
# 値の範囲を -2~2に変更
x = x * 4 - 2
# y=3x-2
y = 3 * x - 2
# 標準正規分布(平均 0, 標準偏差 1の乱数を加える)
y += np.random.randn(100, 1)

model = linear_model.LinearRegression()
model.fit(x, y)

plt.scatter(x, y, marker='+')
plt.scatter(x, model.predict(x), marker='o')
plt.show()

# R2決定係数の計算
r2 = model.score(x, y)

print(model.coef_)
print(model.intercept_)
print(r2)