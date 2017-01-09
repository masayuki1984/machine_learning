import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, svm, ensemble, neighbors

# 0~1までの乱数を1000個作る
x = np.random.rand(1000, 1)
# 値の範囲を -10~10 に変更
x = x * 20 - 10

# 正弦波カープ
y = np.array([math.sin(v) for v in x])
# 標準正規分布(平均 0, 標準偏差 1)の乱数を加える
y += np.random.randn(1000)

# 最小二乗法で解く
model = linear_model.LinearRegression()
model.fit(x, y)

print("最小二乗法: ", model.score(x, y))

plt.scatter(x, y, marker='+')
plt.scatter(x, model.predict(x), marker='o')
plt.show()

# サポートベクターマシンで解く
model = svm.SVR()
model.fit(x, y)

print("SVM: ", model.score(x, y))

plt.scatter(x, y, marker='+')
plt.scatter(x, model.predict(x), marker='o')
plt.show()

# Random Forestで解く
model = ensemble.RandomForestRegressor()
model.fit(x, y)

print("Random Forest: ", model.score(x, y))

plt.scatter(x, y, marker='+')
plt.scatter(x, model.predict(x), marker='o')
plt.show()

# k-nearest neighborで解く
model = neighbors.KNeighborsRegressor()
model.fit(x, y)

print("k-nearest neighbor: ", model.score(x, y))

plt.scatter(x, y, marker='+')
plt.scatter(x, model.predict(x), marker='o')
plt.show()