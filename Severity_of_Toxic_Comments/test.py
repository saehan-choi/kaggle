# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import Ridge, LinearRegression
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.base import TransformerMixin, BaseEstimator
# import re 
# import scipy
# from scipy import sparse
# import gc 
# from IPython.display import display, HTML
# from pprint import pprint
# import warnings
# warnings.filterwarnings("ignore")


# pd.set_option('display.max_columns', 20)
# pd.set_option('display.max_rows', 20)


# df = pd.read_csv("input/train.csv")
# print(df.shape)

# # for col in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
# # #     print(f'****** {col} *******')
# #     display(df.loc[df[col]==1,['comment_text',col]].sample(10))

# df['severe_toxic'] = df.severe_toxic * 2
# df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) ).astype(int)
# df['y'] = df['y']/df['y'].max()
# # sum of y / 7 이랑 똑같음

# print(df['y'].max())

# # print(df.shape)
# # print(df.sample(20))

# p1 = np.array([[5.523,1.2434],
#                 [5.322,9.3435]]).mean(axis=1)

# p2 = np.array([[5.322,9.3435],
#                 [5.523,1.2434]]).mean(axis=1)

# # AXIS가 0이면 열방향에서 자르고
# # AXIS가 1이면 행방향에서 자르네

# print(p1)
# print(p2)

# print(np.round((p1 < p2).mean() * 100,2))

# # round 이건 모르겠네

# p1 = np.array([1,1,1,1,1,1,1,1,1,1])
# p2 = np.array([4,3,2,3,5,6,1,3,8,8])
# print(p1)
# print(p2)

# print(np.round((p1 < p2).mean() * 100,2))

# 서로 True False 비교를하네

# p1 = np.array([True,True,False,False,False])
# print(p1.mean())
# a = np.array([True,False,True,True,False])
# b = np.array([False,True,False,True,False])
# print((a<b).astype('int'))
# [0 1 0 0 0]


# df_sub = pd.read_csv("jigsaw-toxic-severity-rating/comments_to_score.csv")

# print(f'df_sub shape : {df_sub}')

# print(f'count:{df_sub.count()}')

# print(f'nunique:{df_sub.nunique()}')



# # print(df_sub)

# print(df_sub['comment_id'])

# print(df_sub.shape)

# same_score = df_sub['comment_id'].value_counts().reset_index()

# print(same_score)
# print(same_score.shape)
# # print(f'count: {df_sub['score'].count()}')


# z = 351.135134315315
# print(np.around(z ,))

from scipy.stats import rankdata
import numpy as np

array = np.array([1,8,5,7,9,233,6,34,444444,444444,444444,444445])

ranks = rankdata(array)

print(array)
print(ranks)

