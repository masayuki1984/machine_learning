import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cross_validation
import sklearn.svm

# 四国電力の電力消費量データを読み込み
ed = [pd.read_csv(
    'shikoku_electricity_%d.csv' % year,
    skiprows=3,
    names=['DATE', 'TIME', 'consumption'],
    parse_dates={'date_hour': ['DATE', 'TIME']},
    index_col='date_hour')
    for year in [2012, 2013,2014, 2015, 2016]
]

elec_date = pd.concat(ed)

# 気象データを読み込み
tmp = pd.read_csv(
    u'47891_高松.csv',
    parse_dates={'date_hour': ["日時"]},
    index_col="date_hour",
    na_values="X"
)

# [時]の列は使わないので削除
del tmp["時"]

# 列の名前に日本語が入っているとよくないので、これから使う列の名前のみ英語に変更
columns = {
    "降水量(mm)": "rain",
    "気温(℃)": "temperature",
    "日照時間(h)": "sunhour",
    "湿度(%)": "humid"
}
tmp.rename(columns=columns, inplace=True)

# 気象データと電力消費量データを一旦統合して時間軸を合わせた上で、再度分割
takamatsu = elec_date.join(tmp["temperature"]).dropna().as_matrix()

takamatsu_elec = takamatsu[:, 0:1]
takamatsu_wthr = takamatsu[:, 1:]

# 学習と性能の評価
data_count = len(takamatsu_elec)

# 交差検証の準備
kf = sklearn.cross_validation.KFold(data_count, n_folds=5)
print("kf: ", type(kf))

# 交差検証実施(すべてのパターンを実施)
for train, test in kf:
    x_train = takamatsu_wthr[train]
    x_test = takamatsu_wthr[test]
    y_train = takamatsu_elec[train]
    y_test = takamatsu_elec[test]

    # -- SVR --
    model = sklearn.svm.SVR()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    model.fit(x_train, y_train)
    print("Linear: Training Score = %f, Testing(Validate) Score = %f"
              % (model.score(x_train, y_train), model.score(x_test, y_test)))

