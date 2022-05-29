# Libraries
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from glob import glob
from PIL import Image

import cv2

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")

# width , height 순이네요 !
# train\case19\case19_day18\scans\slice_0067_360_310_1.50_1.50
# you can see it's not right width_height_pixelblabla

class CFG:
    seed  = 42
    batch_size = 64
    img_resize = (256, 256)
    # 지금 244로 테스트하는중입니다.
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(img_resize)
                                    ])
                        # if you start with opencv you have to do ToTenser -> resize sequence
# CONSTANTS

pd.set_option('display.max_columns', 500)

train_dir = "./UW-Madison_GI_Tract_Image_Segmentation/input/uw-madison-gi-tract-image-segmentation/train"
training_metadata_path = "./UW-Madison_GI_Tract_Image_Segmentation/input/uw-madison-gi-tract-image-segmentation/train.csv"

# loading metadata
train_df = pd.read_csv(training_metadata_path)
# print(train_df.head())

# train_df['id'] == case123_day20_slice_0001

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

train_df["width"] = train_df["path"].apply(lambda x: os.path.split(x)[-1].split("_")[2]).astype("int")
train_df["height"] = train_df["path"].apply(lambda x: os.path.split(x)[-1].split("_")[3]).astype("int")
# 주최측 실수 -> height, width순이 아니고 width, height 순 입니다.

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

def prepare_mask_data(string):
    # fetching all the values from the string
    all_values = map(int, string.split(" "))
    # preparing the usable arrays
    starterIndex, pixelCount = [], []
    for index, value in enumerate(all_values):
        if index % 2:
            # storing even indexed values in pixelCount
            pixelCount.append(value)
        else:
            # storing odd indexed values in starterIndex
            starterIndex.append(value)
    # 개별로 나눔 ㄷㄷ
    return starterIndex, pixelCount

def fetch_pos_pixel_indexes(indexes, counts):
    final_arr=[]
    for index, counts in zip(indexes, counts):
        final_arr+=[index +i for i in range(counts)]
        # append와 같은역할함 
        # final_arr에는 masked position들이 들어가있습니다. 해석해보면 쉬움

    return final_arr

def prepare_mask(string, height, width):
    # preparing the respective arrays
    indexes, counts = prepare_mask_data(string)
    # preparing all the pixel indexes those have mask values
    pos_pixel_indexes = fetch_pos_pixel_indexes(indexes, counts)
    # forming the flattened array
    mask_array = np.zeros(height * width)
    # updating values in the array
    mask_array[pos_pixel_indexes] = 1
    # reshaping the masks
    return mask_array.reshape(height, width)

def load_image(path):
    # loading the image in RGB format
    image = cv2.imread(path)
    # print(image.shape)
    # image2 = Image.open(path).convert('RGB')
    # !!!!!!! 이거 RGB로 안하면 어떻게 되는지 살펴보기 !!!!!!! --> original image channel is one but we have to change RGB for segmentation tasks
    # print(f'img1:{image.shape}')
    # print(f'img2:{np.array(image2).shape}')
    return image


class UWDataset(Dataset):
    def __init__(self, meta_df):
        super().__init__()
        self.meta_df = meta_df

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, index):
        path = self.meta_df.loc[index, "path"]
        image = load_image(path)
        mask_h, mask_w = self.meta_df.loc[index, "height"], self.meta_df.loc[index, "width"]
        mask_string = self.meta_df.loc[index, "segmentation"]

        main_mask_channel = self.load_mask(string=mask_string, h=mask_h, w=mask_w)
        # 여기엔 정상적으로 저장됨 resize 되기 전
        image = CFG.transform(image)
        main_mask_channel = CFG.transform(main_mask_channel)
        mask = torch.zeros((3, CFG.img_resize[0], CFG.img_resize[1]))
        # resize 이미지 zeros로 가져오기
        class_label = self.meta_df.loc[index, "class"]
        mask[class_label, ...] = main_mask_channel


        return image, mask
        # 이거 plot 해보니깐 이미지들 번진거 좀 있네요 (사이즈조절시 에러난듯.)

    def load_mask(self, string, h, w):
        # cheking if the segmentation encoding is a valid mask or null values
        if string != "nan":
            return Image.fromarray(prepare_mask(string, h, w))
        return Image.fromarray(np.zeros((h, w)))

ds = UWDataset(train_df)
print(f"Length of the dataset : {len(ds)}")


# combined_im_mask = torch.cat([image, mask], dim=2)
# print(combined_im_mask.size()) --> torch.Size([3, 256, 512])

def show_image(tensor_image, name):
    plt.figure(figsize=(10, 10))
    plt.imshow(tensor_image.permute(1,2,0))
    plt.title(name, size=15)
    plt.show()

# image, mask = ds[194]
# print(f"image:{image}")
# print(f"mask:{mask}")
# add_im_mask = torch.add(image, mask)

# print(add_im_mask.size())
# show_image(add_im_mask, "Real & Mask")


train_size = int(len(ds)*0.8)
val_size = len(ds) - train_size
train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(CFG.seed))
print(f"Length of the training dataset : {len(train_ds)}")
print(f"Length of the validation dataset : {len(val_ds)}")


train_dl = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle = True, drop_last = True)
val_dl = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=True, drop_last = True)

for train_image_batch, train_mask_batch in train_dl:
    
    train_image_sample = train_image_batch[0] 
    train_mask_sample = train_mask_batch[0]
    train_batch = torch.add(train_image_sample, train_mask_sample)
    print(train_batch.size())
    show_image(train_batch, "Training Batch")

    # up_date