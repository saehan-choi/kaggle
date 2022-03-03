# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('./input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px

data = pd.read_csv('./input/train.csv')
# print(data['time_step'].describe())

plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), cmap='cool')
# plt.show()

train = data.copy()
bins = [-0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
        2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# print(len(bins))
# print(len(labels))
# print(train)
# train['time_step'] = pd.cut(x=train['time_step'], bins=[-0.99, 0.5, 1, 1.5, 2, 2.5, 3], labels=[0,1,2,3,4,5])
# train['time_step'] = train['time_step'].astype('int64')
# print(train)
# print('a')
# fig, ax = plt.subplots(1,2, figsize=(12,5))
# sns.distplot(train['pressure'], ax=ax[0])

# train['pressure'] = np.where(train['pressure'] < 0, 0, train['pressure'])
# train['pressure'] = np.sqrt(np.sqrt(train['pressure']))
# print('l')
# sns.distplot(train['pressure'], ax=ax[1])
# print('kk')
# plt.show()
# # plt.show()

def create_new_feat(df):
    df["u_in_sum"] = df.groupby("breath_id")["u_in"].transform("sum")
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()
    # cumsum-> sum을 누산
    df["u_in_std"] = df.groupby("breath_id")["u_in"].transform("std")
    df["u_in_min"] = df.groupby("breath_id")["u_in"].transform("min")
    df["u_in_max"] = df.groupby("breath_id")["u_in"].transform("max")
    df["u_in_cumsum_reverse"] = df["u_in_sum"] - df["u_in_cumsum"]

    df["u_in_first"] = df.groupby("breath_id")["u_in"].transform("first")
    df["u_in_last"]  = df.groupby("breath_id")["u_in"].transform("last")

    df['time_diff']  = (df['time_step']).groupby(df['breath_id']).diff(1)
    # diff(1) -> 0번째(지금현재수보다 위에것)와 1번째의차이
    # diff(2) -> 0번째(지금현재수보다 위에것)와 2번째의 차이
    # diff(2) -> 0번째(지금현재수보다 위에것)와 3번째의 차이
    df['time_diff2'] = (df['time_step']).groupby(df['breath_id']).diff(2)
    df['time_diff3'] = (df['time_step']).groupby(df['breath_id']).diff(3)
    df['time_diff4'] = (df['time_step']).groupby(df['breath_id']).diff(4)
    df['time_diff5'] = (df['time_step']).groupby(df['breath_id']).diff(5)
    df['time_diff6'] = (df['time_step']).groupby(df['breath_id']).diff(6)
    df['time_diff7'] = (df['time_step']).groupby(df['breath_id']).diff(7)
    df['time_diff8'] = (df['time_step']).groupby(df['breath_id']).diff(8)
    
    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
    # u_in 한단계 이전 값를 가져옴
    df["u_in_lead1"] = df.groupby("breath_id")["u_in"].shift(-1)
    # u_in 한단계 이후 값를 가져옴
    df["u_in_lag1_diff"] = df["u_in"] - df["u_in_lag1"]
    # 이전값과 현재값의 차이
    # df["u_in_lead1_diff"] = df["u_in"] - df["u_in_lead1"]
    # 현재값과 이후값의 차이 
    df["time_passed"] = df.groupby("breath_id")["time_step"].diff(1)
    # 현재 time_step에서 이전값을 뺀값 반환 (항상양수)

    df['u_in_lag1']  = df.groupby('breath_id')['u_in'].shift(1)
    df['u_in_lag2']  = df.groupby('breath_id')['u_in'].shift(2)
    df['u_in_lag3']  = df.groupby('breath_id')['u_in'].shift(3)
    df['u_in_lag4']  = df.groupby('breath_id')['u_in'].shift(4)
    df['u_in_lag5']  = df.groupby('breath_id')['u_in'].shift(5)
    df['u_in_lag6']  = df.groupby('breath_id')['u_in'].shift(6)
    df['u_in_lag7']  = df.groupby('breath_id')['u_in'].shift(7)
    df['u_in_lag8']  = df.groupby('breath_id')['u_in'].shift(8)
    df['u_in_lag9']  = df.groupby('breath_id')['u_in'].shift(9)
    df['u_in_lag10'] = df.groupby('breath_id')['u_in'].shift(10)
    df['u_in_lag11'] = df.groupby('breath_id')['u_in'].shift(11)
    df['u_in_lag12'] = df.groupby('breath_id')['u_in'].shift(12)
    df['u_in_lag13'] = df.groupby('breath_id')['u_in'].shift(13)
    df['u_in_lag14'] = df.groupby('breath_id')['u_in'].shift(14)
    df['u_in_lag15'] = df.groupby('breath_id')['u_in'].shift(15)
    df['u_in_lag16'] = df.groupby('breath_id')['u_in'].shift(16)
    df['u_in_lag17'] = df.groupby('breath_id')['u_in'].shift(17)
    df['u_in_lag18'] = df.groupby('breath_id')['u_in'].shift(18)
    df['u_in_lag19'] = df.groupby('breath_id')['u_in'].shift(19)
    df['u_in_lag20'] = df.groupby('breath_id')['u_in'].shift(20)
    df['u_in_lag21'] = df.groupby('breath_id')['u_in'].shift(21)
    
    return df

train = create_new_feat(train)
# print(train.isnull().values.any())
# 영향안주도록 0으로 채우면 안되나.. ? 흠
# -> 0으로 해버리면 앞뒤간격을 알수없는듯 더 깊게해보장
train = train.fillna(train.min())
print(train.head())