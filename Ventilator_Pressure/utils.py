import numpy as np
import pandas as pd
import torch

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

def change_columns(value, data):
    change_columns = pd.get_dummies(data[f'{value}'])
    if value == 'C':
        change_columns.rename(columns = {10:f'{value}_10',
                                        20:f'{value}_20',
                                        50:f'{value}_50'}, inplace=True)
    elif value =='R':
        change_columns.rename(columns = {5:f'{value}_5',
                                        20:f'{value}_20',
                                        50:f'{value}_50'}, inplace=True)
    return change_columns

def numpy_to_tensor(variable):
    x = variable.values
    x = np.array(x, dtype=np.float32)
    x = torch.from_numpy(x)
    return x