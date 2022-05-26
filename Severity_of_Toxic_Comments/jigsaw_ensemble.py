import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
import re 
import scipy
from scipy import sparse
import gc 
from IPython.display import display, HTML
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")


# https://www.kaggle.com/andrej0marinchenko/jigsaw-ensemble-0-86

pd.options.display.max_colwidth=300

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("./input/train.csv")
# display(df[:10].sample(10))

for col in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    print(f'****** {col} *******')
    # display(df.loc[df[col]==1,['comment_text',col]].sample(10))
    # df.loc(raw,columns)
    # "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
    # 이런식의 구조를 가짐

# Give more weight to severe toxic 

# print(df.loc[df['severe_toxic']==1, 'severe_toxic'])
# 이거 여기서 severe_toxic의 value를 2배로 만듬
df['severe_toxic'] = df.severe_toxic * 2


df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis=1) ).astype(int)
# print(df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])
df['y'] = df['y']/df['y'].max()

# sum of y / 7 이랑 똑같음

# print(df['y'].sample(10))
# 데이터구조가 onehot encoding으로 이루어진게 아니라, 글안에 'severe_toxic', 'obscene', 'threat', 'insult'
# 이 4가지가 동시에 담길 수 있기때문에 max로 나눠줌(비율을 구함)

# print(df[['comment_text', 'y']])

# print(df.sample(10))
# print(df.shape)
# (159571, 9)

df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})

# print(df.shape)
# (159571, 2)

# print(df['y'].value_counts())

# Create 3 versions of the data
# 여기서부터 할 것
# https://www.kaggle.com/andrej0marinchenko/jigsaw-ensemble-0-86


n_folds = 7
frac_1 = 0.7
frac_1_factor = 1.5

for fld in range(n_folds):
    print(f'Fold: {fld}')
    tmp_df = pd.concat([df[df.y>0].sample(frac=frac_1, random_state = 10*(fld+1)) , 
                        df[df.y==0].sample(n=int(len(df[df.y>0])*frac_1*frac_1_factor) , 
                                            random_state = 10*(fld+1))], axis=0).sample(frac=1, random_state = 10*(fld+1))

    tmp_df.to_csv(f'output/df_fld{fld}.csv', index=False)
    print(tmp_df.shape)
    print(tmp_df['y'].value_counts())

# --------------------------------------------------------------- 2022/01/16

import nltk
# nltk.download('stopwords')
# nltk stopwords 다운로드 안했으면 이거 다운로드부터 먼저해야함
from nltk.corpus import stopwords

stop = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in text]

    # https://wikidocs.net/21707

def clean(data, col):
    
    data[col] = data[col].str.replace('https?://\S+|www\.\S+', ' social medium ')      
        
    data[col] = data[col].str.lower()
    # 소문자로 변경해줌
    data[col] = data[col].str.replace("4", "a") 
    data[col] = data[col].str.replace("2", "l")
    data[col] = data[col].str.replace("5", "s") 
    data[col] = data[col].str.replace("1", "i") 
    data[col] = data[col].str.replace("!", "i") 
    data[col] = data[col].str.replace("|", "i") 
    data[col] = data[col].str.replace("0", "o") 
    data[col] = data[col].str.replace("l3", "b") 
    data[col] = data[col].str.replace("7", "t") 
    data[col] = data[col].str.replace("7", "+") 
    data[col] = data[col].str.replace("8", "ate") 
    data[col] = data[col].str.replace("3", "e") 
    data[col] = data[col].str.replace("9", "g")
    data[col] = data[col].str.replace("6", "g")
    data[col] = data[col].str.replace("@", "a")
    data[col] = data[col].str.replace("$", "s")
    data[col] = data[col].str.replace("#ofc", " of fuckin course ")
    data[col] = data[col].str.replace("fggt", " faggot ")
    # 줄임말이나, 띄어쓰기를 다시 고쳐주네여
    data[col] = data[col].str.replace("your", " your ")
    data[col] = data[col].str.replace("self", " self ")
    data[col] = data[col].str.replace("cuntbag", " cunt bag ")
    data[col] = data[col].str.replace("fartchina", " fart china ")    
    data[col] = data[col].str.replace("youi", " you i ")
    data[col] = data[col].str.replace("cunti", " cunt i ")
    data[col] = data[col].str.replace("sucki", " suck i ")
    data[col] = data[col].str.replace("pagedelete", " page delete ")
    data[col] = data[col].str.replace("cuntsi", " cuntsi ")
    data[col] = data[col].str.replace("i'm", " i am ")
    data[col] = data[col].str.replace("offuck", " of fuck ")
    data[col] = data[col].str.replace("centraliststupid", " central ist stupid ")
    data[col] = data[col].str.replace("hitleri", " hitler i ")
    data[col] = data[col].str.replace("i've", " i have ")
    data[col] = data[col].str.replace("i'll", " sick ")
    data[col] = data[col].str.replace("fuck", " fuck ")
    data[col] = data[col].str.replace("f u c k", " fuck ")
    data[col] = data[col].str.replace("shit", " shit ")
    data[col] = data[col].str.replace("bunksteve", " bunk steve ")
    data[col] = data[col].str.replace('wikipedia', ' social medium ')
    data[col] = data[col].str.replace("faggot", " faggot ")
    data[col] = data[col].str.replace("delanoy", " delanoy ")
    data[col] = data[col].str.replace("jewish", " jewish ")
    data[col] = data[col].str.replace("sexsex", " sex ")
    data[col] = data[col].str.replace("allii", " all ii ")
    data[col] = data[col].str.replace("i'd", " i had ")
    data[col] = data[col].str.replace("'s", " is ")
    data[col] = data[col].str.replace("youbollocks", " you bollocks ")
    data[col] = data[col].str.replace("dick", " dick ")
    data[col] = data[col].str.replace("cuntsi", " cuntsi ")
    data[col] = data[col].str.replace("mothjer", " mother ")
    data[col] = data[col].str.replace("cuntfranks", " cunt ")
    data[col] = data[col].str.replace("ullmann", " jewish ")
    data[col] = data[col].str.replace("mr.", " mister ")
    data[col] = data[col].str.replace("aidsaids", " aids ")
    data[col] = data[col].str.replace("njgw", " nigger ")
    data[col] = data[col].str.replace("wiki", " social medium ")
    data[col] = data[col].str.replace("administrator", " admin ")
    data[col] = data[col].str.replace("gamaliel", " jewish ")
    data[col] = data[col].str.replace("rvv", " vanadalism ")
    data[col] = data[col].str.replace("admins", " admin ")
    data[col] = data[col].str.replace("pensnsnniensnsn", " penis ")
    data[col] = data[col].str.replace("pneis", " penis ")
    data[col] = data[col].str.replace("pennnis", " penis ")
    data[col] = data[col].str.replace("pov.", " point of view ")
    data[col] = data[col].str.replace("vandalising", " vandalism ")
    data[col] = data[col].str.replace("cock", " dick ")
    data[col] = data[col].str.replace("asshole", " asshole ")
    data[col] = data[col].str.replace("youi", " you ")
    data[col] = data[col].str.replace("afd", " all fucking day ")
    data[col] = data[col].str.replace("sockpuppets", " sockpuppetry ")
    data[col] = data[col].str.replace("iiprick", " iprick ")
    data[col] = data[col].str.replace("penisi", " penis ")
    data[col] = data[col].str.replace("warrior", " warrior ")
    data[col] = data[col].str.replace("loil", " laughing out insanely loud ")
    data[col] = data[col].str.replace("vandalise", " vanadalism ")
    data[col] = data[col].str.replace("helli", " helli ")
    data[col] = data[col].str.replace("lunchablesi", " lunchablesi ")
    data[col] = data[col].str.replace("special", " special ")
    data[col] = data[col].str.replace("ilol", " i lol ")
    data[col] = data[col].str.replace(r'\b[uU]\b', 'you')
    data[col] = data[col].str.replace(r"what's", "what is ")
    data[col] = data[col].str.replace(r"\'s", " is ")
    data[col] = data[col].str.replace(r"\'ve", " have ")
    data[col] = data[col].str.replace(r"can't", "cannot ")
    data[col] = data[col].str.replace(r"n't", " not ")
    data[col] = data[col].str.replace(r"i'm", "i am ")
    data[col] = data[col].str.replace(r"\'re", " are ")
    data[col] = data[col].str.replace(r"\'d", " would ")
    data[col] = data[col].str.replace(r"\'ll", " will ")
    data[col] = data[col].str.replace(r"\'scuse", " excuse ")
    data[col] = data[col].str.replace('\s+', ' ')  # will remove more than one whitespace character
#     text = re.sub(r'\b([^\W\d_]+)(\s+\1)+\b', r'\1', re.sub(r'\W+', ' ', text).strip(), flags=re.I)  # remove repeating words coming immediately one after another
    data[col] = data[col].str.replace(r'(.)\1+', r'\1\1') # 2 or more characters are replaced by 2 characters
#     text = re.sub(r'((\b\w+\b.{1,2}\w+\b)+).+\1', r'\1', text, flags = re.I)
    data[col] = data[col].str.replace("[:|♣|'|§|♠|*|/|?|=|%|&|-|#|•|~|^|>|<|►|_]", '')
    
    
    data[col] = data[col].str.replace(r"what's", "what is ")    
    data[col] = data[col].str.replace(r"\'ve", " have ")
    data[col] = data[col].str.replace(r"can't", "cannot ")
    data[col] = data[col].str.replace(r"n't", " not ")
    data[col] = data[col].str.replace(r"i'm", "i am ")
    data[col] = data[col].str.replace(r"\'re", " are ")
    data[col] = data[col].str.replace(r"\'d", " would ")
    data[col] = data[col].str.replace(r"\'ll", " will ")
    data[col] = data[col].str.replace(r"\'scuse", " excuse ")
    data[col] = data[col].str.replace(r"\'s", " ")

    # Clean some punctutations
    data[col] = data[col].str.replace('\n', ' \n ')
    data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3')
    # Replace repeating characters more than 3 times to length of 3
    data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1')    
    # Add space around repeating characters
    data[col] = data[col].str.replace(r'([*!?\']+)',r' \1 ')    
    # patterns with repeating characters 
    data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1')
    data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1')
    data[col] = data[col].str.replace(r'[ ]{2,}',' ').str.strip()   
    data[col] = data[col].str.replace(r'[ ]{2,}',' ').str.strip()   
    data[col] = data[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    return data


# Test clean function
# test_clean_df = pd.DataFrame({"text":
#                               ["heyy\n\nkkdsfj",
#                                "hi   how/are/you ???",
#                                "hey?????",
#                                "noooo!!!!!!!!!   comeone !! ",
#                               "cooooooooool     brooooooooooo  coool brooo",
#                               "naaaahhhhhhh"]})
# display(test_clean_df)
# display(clean(test_clean_df,'text'))
# 예시테스트 해볼수있음

df = clean(df,'text')
# display(df[df.y==0].sample(10))
# 위에 꽤 걸리는 작업이네여

n_folds = 7

frac_1 = 0.7
frac_1_factor = 1.5

for fld in range(n_folds):
    tmp_df = pd.concat([df[df.y>0].sample(frac=frac_1, random_state = 10*(fld+1)) , 
                        df[df.y==0].sample(n=int(len(df[df.y>0])*frac_1*frac_1_factor) , 
                                            random_state = 10*(fld+1))], axis=0).sample(frac=1, random_state = 10*(fld+1))

# frac : 0.5 일시 50%데이터를 가져옴 random_state에 따라 다름
# n : n개의 숫자만큼 데이터를 가져옴 random_state에 따라 다름

    tmp_df.to_csv(f'output/df_clean_fld{fld}.csv', index=False)
    print(tmp_df.shape)
    print(tmp_df['y'].value_counts())


del df,tmp_df
gc.collect()
# 메모리해제

# kaggle에 ruddit-jigsaw-dataset 검색하면 데이터나옴
df_ = pd.read_csv("ruddit-jigsaw-dataset/ruddit_with_text.csv")
clean(df_,'txt')
print(df_.shape)
print(df_)
df_ = df_[['txt', 'offensiveness_score']].rename(columns={'txt': 'text',
                                                                'offensiveness_score':'y'})

df_['y'] = (df_['y'] - df_.y.min()) / (df_.y.max() - df_.y.min()) 
# 정규화과정, 안에보면 score(y)가 -까지 분포하는걸로 보아 0~1사이 값을 만들어줌
# df_.y.hist()

n_folds = 7

frac_1 = 0.7

for fld in range(n_folds):
    print(f'Fold: {fld}')
    tmp_df = df_.sample(frac=frac_1, random_state = 10*(fld+1))
    tmp_df.to_csv(f'output/df2_fld{fld}.csv', index=False)
    print(tmp_df.shape)
    print(tmp_df['y'].value_counts())

del tmp_df, df_
gc.collect()


# Validation data 

df_val = pd.read_csv("jigsaw-toxic-severity-rating/validation_data.csv")
# clean(df_val,'less_toxic')
# clean(df_val,'more_toxic')
# print(df_val)


df_sub = pd.read_csv("jigsaw-toxic-severity-rating/comments_to_score.csv")
# clean(df_sub,'text')
print(df_sub)


# ----------------------------------- 여기까지 feature engneering  -----------

# https://www.kaggle.com/andrej0marinchenko/jigsaw-ensemble-0-86
# Create Sklearn Pipeline with 여기부터하면됨


val_preds_arr1 = np.zeros((df_val.shape[0], n_folds))
val_preds_arr2 = np.zeros((df_val.shape[0], n_folds))
test_preds_arr = np.zeros((df_sub.shape[0], n_folds))

for fld in range(n_folds):
    print("\n\n")
    print(f' ****************************** FOLD: {fld} ******************************')
    df = pd.read_csv(f'output/df_fld{fld}.csv')
    print(df.shape)

    features = FeatureUnion([
        #('vect1', LengthTransformer()),
        #('vect2', LengthUpperTransformer()),
        ("vect3", TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5))),
        # char -> 글자수로 짜름 ngram은 3~5의 글자를 짜름  ex) bed, airpl, ane 등
        # analyzer = 'word' -> 한 단어를 기준으로 짜름 ex) bed, room, office
        # min_df -> document-frequency 특정단어가 나타나는 문서의 수(df = 3 이면 3개의 배열에 그 단어가 300번나타나도 df =3 임)
        #("vect4", TfidfVectorizer(min_df= 5, max_df=0.5, analyzer = 'word', token_pattern=r'(?u)\b\w{8,}\b')),
    ])
    
    pipeline = Pipeline(
        [
            ("features", features),
            #("clf", RandomForestRegressor(n_estimators = 5, min_sample_leaf=3)),
            ("clf", Ridge()),
            #("clf",LinearRegression())
        ]
    )
    # clf : classifier

    print("\nTrain:")
    # Train the pipeline
    pipeline.fit(df['text'], df['y'])
    
    # What are the important features for toxicity

    print('\nTotal number of features:', len(pipeline['features'].get_feature_names()) )

    feature_wts = sorted(list(zip(pipeline['features'].get_feature_names(), 
                                  np.round(pipeline['clf'].coef_,2) )), 
                         key = lambda x:x[1], 
                         reverse=True)

    pprint(feature_wts[:100])
    # 100개씩 보겠다
    
    print("\npredict validation data ")
    val_preds_arr1[:,fld] = pipeline.predict(df_val['less_toxic'])
    val_preds_arr2[:,fld] = pipeline.predict(df_val['more_toxic'])

    print("\npredict test data ")
    test_preds_arr[:,fld] = pipeline.predict(df_sub['text'])

val_preds_arr1c = np.zeros((df_val.shape[0], n_folds))
val_preds_arr2c = np.zeros((df_val.shape[0], n_folds))
test_preds_arrc = np.zeros((df_sub.shape[0], n_folds))

for fld in range(n_folds):
    print("\n\n")
    print(f' ****************************** FOLD: {fld} ******************************')
    df = pd.read_csv(f'output/df_clean_fld{fld}.csv')
    print(df.shape)

    features = FeatureUnion([
        #('vect1', LengthTransformer()),
        #('vect2', LengthUpperTransformer()),
        ("vect3", TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5))),
        #("vect4", TfidfVectorizer(min_df= 5, max_df=0.5, analyzer = 'word', token_pattern=r'(?u)\b\w{8,}\b')),

    ])
    pipeline = Pipeline(
        [
            ("features", features),
            #("clf", RandomForestRegressor(n_estimators = 5, min_sample_leaf=3)),
            ("clf", Ridge()),
            #("clf",LinearRegression())
        ]
    )
    print("\nTrain:")
    # Train the pipeline
    pipeline.fit(df['text'].values.astype('U'), df['y'].values.astype('U'))
    
    # What are the important features for toxicity

    print('\nTotal number of features:', len(pipeline['features'].get_feature_names()) )

    feature_wts = sorted(list(zip(pipeline['features'].get_feature_names(), 
                                  np.round(pipeline['clf'].coef_,2) )), 
                         key = lambda x:x[1], 
                         reverse=True)

    pprint(feature_wts[:30])
    
    print("\npredict validation data ")
    val_preds_arr1c[:,fld] = pipeline.predict(df_val['less_toxic'])
    val_preds_arr2c[:,fld] = pipeline.predict(df_val['more_toxic'])

    print("\npredict test data ")
    test_preds_arrc[:,fld] = pipeline.predict(df_sub['text'])


val_preds_arr1_ = np.zeros((df_val.shape[0], n_folds))
val_preds_arr2_ = np.zeros((df_val.shape[0], n_folds))
test_preds_arr_ = np.zeros((df_sub.shape[0], n_folds))

for fld in range(n_folds):
    print("\n\n")
    print(f' ****************************** FOLD: {fld} ******************************')
    df = pd.read_csv(f'output/df2_fld{fld}.csv')
    print(df.shape)

    features = FeatureUnion([
        #('vect1', LengthTransformer()),
        #('vect2', LengthUpperTransformer()),
        ("vect3", TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5))),
        #("vect4", TfidfVectorizer(min_df= 5, max_df=0.5, analyzer = 'word', token_pattern=r'(?u)\b\w{8,}\b')),

    ])
    pipeline = Pipeline(
        [
            ("features", features),
            #("clf", RandomForestRegressor(n_estimators = 5, min_sample_leaf=3)),
            ("clf", Ridge()),
            #("clf",LinearRegression())
        ]
    )
    print("\nTrain:")
    # Train the pipeline
    pipeline.fit(df['text'].values.astype('U'), df['y'].values.astype('U'))
    
    # What are the important features for toxicity

    print('\nTotal number of features:', len(pipeline['features'].get_feature_names()) )

    feature_wts = sorted(list(zip(pipeline['features'].get_feature_names(), 
                                  np.round(pipeline['clf'].coef_,2) )), 
                         key = lambda x:x[1], 
                         reverse=True)

    pprint(feature_wts[:30])
    
    print("\npredict validation data ")
    val_preds_arr1_[:,fld] = pipeline.predict(df_val['less_toxic'])
    val_preds_arr2_[:,fld] = pipeline.predict(df_val['more_toxic'])

    print("\npredict test data ")
    test_preds_arr_[:,fld] = pipeline.predict(df_sub['text'])


del df, pipeline, feature_wts
gc.collect()

print(" Toxic data ")
p1 = val_preds_arr1.mean(axis=1)
p2 = val_preds_arr2.mean(axis=1)
# pn에는 이미 값들이 들어가있음 p(1+2d) (d=0,1,2....)의 값들은 각 학습된 데이터기반으로 
# 본래 프로젝트 목표의 score값을 반환

print(f'Validation Accuracy is { np.round((p1 < p2).mean() * 100,2)}')
# np.round((p1 < p2).mean() 이거 뭔지 잘 모르겠음.....
# -> 대소비교함으로서 True,False를 반환, 그리고 mean()을 주어서 평균값을 매김
# p1 = np.array([True,True,False,False,False])
# print(p1.mean())
# --> 0.4

print(" Ruddit data ")
p3 = val_preds_arr1_.mean(axis=1)
p4 = val_preds_arr2_.mean(axis=1)

print(f'Validation Accuracy is { np.round((p3 < p4).mean() * 100,2)}')

print(" Toxic CLEAN data ")
p5 = val_preds_arr1c.mean(axis=1)
p6 = val_preds_arr2c.mean(axis=1)

print(f'Validation Accuracy is { np.round((p5 < p6).mean() * 100,2)}')


print("Find right weight")

wts_acc = []
for i in range(30,70,1):
    for j in range(0,20,1):
        w1 = i/100
        w2 = (100 - i - j)/100
        w3 = (1 - w1 - w2 )
        p1_wt = w1*p1 + w2*p3 + w3*p5
        p2_wt = w1*p2 + w2*p4 + w3*p6
        wts_acc.append((w1,w2,w3, 
                         np.round((p1_wt < p2_wt).mean() * 100,2))
                      )

print(sorted(wts_acc, key=lambda x:x[3], reverse=True)[:5])
# Find right weight
# [(0.67, 0.26, 0.06999999999999995, 69.33),
#  (0.65, 0.25, 0.09999999999999998, 69.31),
#  (0.64, 0.25, 0.10999999999999999, 69.29),
#  (0.65, 0.24, 0.10999999999999999, 69.29),
#  (0.66, 0.26, 0.07999999999999996, 69.29)]

w1,w2,w3,_ = sorted(wts_acc, key=lambda x:x[2], reverse=True)[0]
#print(best_wts)
# 이건 lambda x[3] 에서 x[2]로 바뀌었는데.. ? ?

p1_wt = w1*p1 + w2*p3 + w3*p5
p2_wt = w1*p2 + w2*p4 + w3*p6




df_val['p1'] = p1_wt
df_val['p2'] = p2_wt
df_val['diff'] = np.abs(p2_wt - p1_wt)

df_val['correct'] = (p1_wt < p2_wt).astype('int')

# a = np.array([True,False,True,True,False])
# b = np.array([False,True,False,True,False])
# print((a<b).astype('int'))
# [0 1 0 0 0]

### Incorrect predictions with similar scores

df_val[df_val.correct == 0].sort_values('diff', ascending=True).head(20)
# 오름차순


df_val[df_val.correct == 0].sort_values('diff', ascending=False).head(20)
# 내림차순


# Predict using pipeline

df_sub['score'] = w1*test_preds_arr.mean(axis=1) + w2*test_preds_arr_.mean(axis=1) + w3*test_preds_arrc.mean(axis=1)

# Cases with duplicates scores (중복점수가 있는경우)
df_sub['score'].count() - df_sub['score'].nunique()
# nunique -> number of unique 고유값의 숫자들을 반환
# 고유값 -> 중복값말고 유일한 값만 있는것
# 이 둘의 차가 적다는말은 중복되는 요소의 갯수가 작다는 말임


same_score = df_sub['score'].value_counts().reset_index()[:10]
print(same_score)
# value_counts 하면 score가 몇개있는지 value count 되고
# 0.2352524 1
# 0.53632 1
# 0.742525 1
# 0.864643 1
# 이런식으로 index가 사라지는데 reset_index를 함으로서 index를 복귀시킴
# 이거하는 이유는 똑같은점수가 몇개있는지 확인하기 위한용


print(df_sub[df_sub['score'].isin(same_score['index'].tolist())])
# https://3months.tistory.com/283
# 여기설명 잘되어있습니다.
# to list를 함으로서 값을가지는것들이 score에 있는것들을 가져옴
# 근데 굳이...? 똑같음 하나 안하나
# -> 아님 같은것들은 같은스코어끼리 정렬이 되네요
# 안에 같은거있는것만 골라내줌

import os
import gc
import cv2
import copy
import time
import random

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For Transformer Models
from transformers import AutoTokenizer, AutoModel

# Utils
from tqdm import tqdm

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# cuda 관련 더 자세한 에러를 보려면 이거 해야함

CONFIG = dict(
    seed = 42,
    model_name = '../input/roberta-base',
    test_batch_size = 64,
    max_length = 128,
    num_classes = 1,
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
)


CONFIG["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG['model_name'])


MODEL_PATHS = [
    '../input/k/saurabhbagchi/pytorch-w-b-jigsaw-starter/Loss-Fold-0.bin',
    '../input/k/saurabhbagchi/pytorch-w-b-jigsaw-starter/Loss-Fold-1.bin',
    '../input/k/saurabhbagchi/pytorch-w-b-jigsaw-starter/Loss-Fold-2.bin',
    '../input/k/saurabhbagchi/pytorch-w-b-jigsaw-starter/Loss-Fold-3.bin',
    '../input/k/saurabhbagchi/pytorch-w-b-jigsaw-starter/Loss-Fold-4.bin'
]


# 여기는 BERT 나옴