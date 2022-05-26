import numpy as np
import pandas as pd
# pd.options.plotting.backend = "plotly"
import random
from glob import glob
import os, shutil
from tqdm import tqdm
tqdm.pandas()
import time
import copy
import joblib
import gc
from IPython import display as ipd
from joblib import Parallel, delayed

# visualization
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''

    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_metadata(row):
    data = row['id'].split('_')
    case = int(data[0].replace('case',''))
    day = int(data[1].replace('day',''))
    slice_ = int(data[-1])

    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row

def path2info(row):
    #                               id        class segmentation  case  day  slice
    # 0       case123_day20_slice_0001  large_bowel          NaN   123   20      1
    # 1       case123_day20_slice_0001  small_bowel          NaN   123   20      1
    # 2       case123_day20_slice_0001      stomach          NaN   123   20      1
    # 3       case123_day20_slice_0002  large_bowel          NaN   123   20      2 ...    
    path = row['image_path']
    data = path.split('\\')
    slice_ = int(data[-1].split('_')[1])
    case = int(data[-3].split('_')[0].replace('case',''))
    day = int(data[-3].split('_')[1].replace('day',''))
    width = int(data[-1].split('_')[2])
    height = int(data[-1].split('_')[3])
    row['height'] = height
    row['width'] = width
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row

def id2mask(id_):
    idf = df[df['id']==id_]
    wh = idf[['height','width']].iloc[0]
    shape = (wh.height, wh.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(['large_bowel', 'small_bowel', 'stomach']):
        cdf = idf[idf['class']==class_]
        rle = cdf.segmentation.squeeze()
        if len(cdf) and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask

def rgb2gray(mask):
    pad_mask = np.pad(mask, pad_width=[(0,0),(0,0),(1,0)])
    gray_mask = pad_mask.argmax(-1)
    return gray_mask

def gray2rgb(mask):
    rgb_mask = tf.keras.utils.to_categorical(mask, num_classes=4)
    return rgb_mask[..., 1:].astype(mask.dtype)

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype('float32') # original is uint16
    img = (img - img.min())/(img.max() - img.min())*255.0 # scale image to [0, 255]
    img = img.astype('uint8')
    return img

def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
#     plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='bone')
    
    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = [ "Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles,labels)
    plt.axis('off')
    # plt.show()

df = pd.read_csv('..\\input\\uw-madison-gi-tract-image-segmentation\\train.csv')
# df = df[~df.segmentation.isna()]
df = df.progress_apply(get_metadata, axis=1)

# print(df.head())
#                               id        class segmentation  case  day  slice
# 0       case123_day20_slice_0001  large_bowel          NaN   123   20      1
# 1       case123_day20_slice_0001  small_bowel          NaN   123   20      1
# 2       case123_day20_slice_0001      stomach          NaN   123   20      1
# 3       case123_day20_slice_0002  large_bowel          NaN   123   20      2 ...



paths = glob('..\\input\\uw-madison-gi-tract-image-segmentation\\train\\*\\*\\*\\*')
# glob을 이용하면 해당 파일들을 list형식으로 반환합니다.
# 지금 데이터 인풋이 바뀌어서 못받아오는겁니다 첨부터 다시 받으십시오.

path_df = pd.DataFrame(paths, columns=['image_path'])
path_df = path_df.progress_apply(path2info, axis=1)
df = df.merge(path_df, on=['case','day','slice'])

print(df.head())
#                               id        class segmentation  case  day  slice                                         image_path  height  width
# 0       case123_day20_slice_0001  large_bowel          NaN   123   20      1  ../input/uw-madison-gi-tract-image-segmentatio...     266    266
# 1       case123_day20_slice_0001  large_bowel          NaN   123   20      1  ../input/uw-madison-gi-tract-image-segmentatio...     266    266
# 2       case123_day20_slice_0001  small_bowel          NaN   123   20      1  ../input/uw-madison-gi-tract-image-segmentatio...     266    266
# 3       case123_day20_slice_0001  small_bowel          NaN   123   20      1  ../input/uw-madison-gi-tract-image-segmentatio...     266    266
# 4       case123_day20_slice_0001      stomach          NaN   123   20      1  ../input/uw-madison-gi-tract-image-segmentatio...     266    266
# ~ -> True -> False로 변경

row=1; col=4
plt.figure(figsize=(5*col,5*row))
for i, id_ in enumerate(df[~df.segmentation.isna()].sample(frac=1.0)['id'].unique()[:row*col]):
    # 4개만 plot 띄어보겠다 이거네 이건ㅋㅋ
    print(f'id_:{id_}')
    print(f"df[df['id']==id_].image_path.iloc[0]:{df[df['id']==id_].image_path.iloc[0]}")

    img = load_img(df[df['id']==id_].image_path.iloc[0])
    mask = id2mask(id_)*255
    plt.subplot(row, col, i+1)
    i+=1
    show_img(img, mask=mask)
    plt.tight_layout() 


# 주석부분부터 다시 볼 것
# def save_mask(id_):
#     idf = df[df['id']==id_]
#     mask = id2mask(id_)*255
#     image_path = idf.image_path.iloc[0]
#     # 첫번째행 가져오기
#     mask_path = image_path.replace('\\input\\','\\tmp\\png\\')
#     mask_folder = mask_path.rsplit('\\',1)[0]
#     print(f'mask_folder:{mask_folder}')
#     os.makedirs(mask_folder, exist_ok=True)
#     cv2.imwrite(mask_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])
#     mask_path2 = image_path.replace('\\input\\','\\tmp\\np\\').replace('.png','.npy')
#     mask_folder2 = mask_path2.rsplit('\\',1)[0]
#     os.makedirs(mask_folder2, exist_ok=True)
#     np.save(mask_path2, mask)
#     return mask_path

# ids = df['id'].unique()
# _ = Parallel(n_jobs=-1, backend='threading')(delayed(save_mask)(id_)\
#                                              for id_ in tqdm(ids, total=len(ids)))

# i = 250
# img = load_img(df.image_path.iloc[i])
# mask_path = df['image_path'].iloc[i].replace('\\input\\','\\tmp\\png\\')
# mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
# plt.figure(figsize=(5,5))
# show_img(img, mask=mask)

# df['mask_path'] = df.image_path.str.replace('\\input','\\input\\uwmgi-mask-dataset\\png\\')
# df.to_csv('train.csv',index=False)

# shutil.make_archive('/kaggle/working/png',
#                     'zip',
#                     '/tmp/png',
#                     'uw-madison-gi-tract-image-segmentation')

# shutil.make_archive('/kaggle/working/np',
#                     'zip',
#                     '/tmp/np',
#                     'uw-madison-gi-tract-image-segmentation')