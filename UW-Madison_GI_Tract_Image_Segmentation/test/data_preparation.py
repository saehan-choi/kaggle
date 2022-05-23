# Libraries
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize

# CONSTANTS
SEED  = 42
BATCH_SIZE = 64
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)

train_dir = "./UW-Madison_GI_Tract_Image_Segmentation/input/uw-madison-gi-tract-image-segmentation/train"
training_metadata_path = "./UW-Madison_GI_Tract_Image_Segmentation/input/uw-madison-gi-tract-image-segmentation/train.csv"

# loading metadata
train_df = pd.read_csv(training_metadata_path)
# print(train_df.head())

train_df["segmentation"] = train_df["segmentation"].astype("str")
train_df["case_id"] = train_df["id"].apply(lambda x: x.split("_")[0][4:])
train_df["day_id"] = train_df["id"].apply(lambda x: x.split("_")[1][3:])
train_df["slice_id"] = train_df["id"].apply(lambda x: x.split("_")[-1])
print(train_df.head())

def fetch_file_from_id(root_dir, case_id):
    case_folder = case_id.split("_")[0]
    day_folder = "_".join(case_id.split("_")[:2])
    # "___".join(case_id.split("_")[:2]) --> case123___day20
    file_starter = "_".join(case_id.split("_")[2:])
    # fetching folder paths
    folder = os.path.join(root_dir, case_folder, day_folder, "scans")
    # fetching filenames with similar pattern
    file = glob(f"{folder}/{file_starter}*")
    # returning the first file, though it will always hold one file. --> it's right haha.
    return file[0]

train_df["path"] = train_df["id"].apply(lambda x: fetch_file_from_id(train_dir, x))
# pandas를 함수에 적용할때는 apply를 이용
# train_dir = "./UW-Madison_GI_Tract_Image_Segmentation/input/uw-madison-gi-tract-image-segmentation/train"
# case_id = case123_day20_slice_0001

train_df["height"] = train_df["path"].apply(lambda x: os.path.split(x)[-1].split("_")[2]).astype("int")
train_df["width"] = train_df["path"].apply(lambda x: os.path.split(x)[-1].split("_")[3]).astype("int")
# os.path.split(x)[-1] --> slice_0001_266_266_1.50_blah_balh 입니다
# train_df.head()
# path = '/home/User/Desktop/1_test/2_test/3_test'
# head_tail = os.path.split(path)
# print(head_tail)  -->  ('/home/User/Desktop/1_test/2_test', '3_test')


class_names = train_df["class"].unique()
# --> ['large_bowel', 'small_bowel', 'stomach']
for index, label in enumerate(class_names):
    # replacing class names with indexes
    train_df["class"].replace(label, index, inplace = True)

# Mask Generation Methodology :
# https://www.kaggle.com/code/sagnik1511/uwmgit-data-preparation-from-scratch
# 여기부터 다시 할 것
