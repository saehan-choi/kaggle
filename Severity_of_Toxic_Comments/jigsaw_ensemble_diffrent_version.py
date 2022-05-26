import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import rankdata


df = pd.read_csv("./input/train_data_version2.csv")
df.shape
df = df[['text', 'y']]
vec = TfidfVectorizer(analyzer='char_wb', max_df=0.7, min_df=1, ngram_range=(2, 5) )
# ngram_range -> 단어 2개가 함께 사용되는것부터 5개단어 사용되는것까지 묶어서 가져오겠다
# char_wb도 글자 기준이긴하네여....


# print(df['text'])
# print(df['text'].shape)


X = vec.fit_transform(df['text'])
print(X)

z = df["y"].values
y=np.around(z ,decimals = 2)

model1=Ridge(alpha=0.5)
model1.fit(X, y)
df_test = pd.read_csv("./jigsaw-toxic-severity-rating/comments_to_score.csv")
print(df_test)
test=vec.transform(df_test['text'])
print(test)
# jr_preds=model1.predict(test)


# df_test['score1']=rankdata( jr_preds, method='ordinal') 
# rud_df = pd.read_csv("./ruddit-jigsaw-dataset/ruddit_with_text.csv")
# #print(f"rud_df:{rud_df.shape}")
# rud_df['y'] = rud_df["offensiveness_score"] 
# df = rud_df[['txt', 'y']].rename(columns={'txt': 'text'})
# vec = TfidfVectorizer(analyzer='char_wb', max_df=0.7, min_df=3, ngram_range=(3, 4) )
# X = vec.fit_transform(df['text'])
# z = df["y"].values
# y=np.around ( z ,decimals = 1)
# model1=Ridge(alpha=0.5)
# model1.fit(X, y)
# test=vec.transform(df_test['text'])
# rud_preds=model1.predict(test)
# df_test['score2']=rankdata( rud_preds, method='ordinal')
# df_test['score']=df_test['score1']+df_test['score2']
# df_test['score']=rankdata( df_test['score'], method='ordinal')
# df_test[['comment_id', 'score']].to_csv("submission1.csv", index=False)


# # https://www.kaggle.com/thomasdubail/jigsaw-ensemble-best-public-sub-0-898 여기에 더있음