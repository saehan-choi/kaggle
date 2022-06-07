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

from unet.unet_model import *
from unet.utils.dice_score import dice_coef, iou_coef

import torch.optim as optim

from tqdm import tqdm

# width , height 순이네요 !
# train\case19\case19_day18\scans\slice_0067_360_310_1.50_1.50
# you can see it's not right width_height_pixelblabla

# https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch
# RESTART!!!!!!!!!!!!!!!!!!!


class CFG:
    seed  = 42
    batch_size = 8
    img_resize = (256, 256)
    device = 'cuda'
    # 지금 244로 테스트하는중입니다.
    transform  = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(img_resize)
                                    ])
                        # if you start with opencv you have to do ToTenser -> resize sequence
    n_channels = 3
    n_classes  = 3
    epochs = 30
    lr = 1e-5


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

    image = Image.open(path).convert("RGB")
    # -> channel 3  but original image shape is 3
    # !!!!!!! 이거 RGB로 안하면 어떻게 되는지 살펴보기 !!!!!!! --> original image channel is one but we have to change RGB for segmentation tasks
    # image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')
    # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # image = image.astype(np.uint8)

    # -> channel 1

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

    def load_mask(self, string, h, w):
        # cheking if the segmentation encoding is a valid mask or null values
        if string != "nan":
            return Image.fromarray(prepare_mask(string, h, w))
        return Image.fromarray(np.zeros((h, w)))



# combined_im_mask = torch.cat([image, mask], dim=2)
# print(combined_im_mask.size()) --> torch.Size([3, 256, 512])

def show_image(tensor_image, name):
    plt.figure(figsize=(5, 5))
    plt.imshow(tensor_image.permute(1,2,0))
    plt.text(0, -30, 'your legend', bbox={'facecolor': 'blue', 'pad': 2})
    plt.text(0, -15, 'your legend', bbox={'facecolor': 'green', 'pad': 2})
    plt.text(0, 0, 'your legend', bbox={'facecolor': 'red', 'pad': 2})
    # large_bowel -> 0
    # small_bowel -> 1 
    # stomach     -> 2

    plt.title(name, size=10)
    plt.show()

# image, mask = ds[194]
# print(f"image:{image}")
# print(f"mask:{mask}")
# add_im_mask = torch.add(image, mask)

# print(add_im_mask.size())
# show_image(add_im_mask, "Real & Mask")


def train_one_epoch(model, optimizer, criterion, dataloader, epoch, device):
    model.train()
    dataset_size = 0
    running_loss = 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        
        images = data[0].to(device, dtype=torch.float)
        # this is float now I don't know it's right.
        mask_true = data[1].to(device, dtype=torch.float)
        batch_size = images.size(0)
        mask_pred = model(images)

        # print(mask_true.to(dtype=torch.long))
        # print(mask_true.permute(0, 3, 1, 2).size())

        loss = criterion(mask_pred, mask_true)        
        dice_coeff = dice_coef(mask_pred, mask_true)

        # is this sequence right? it's right
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(epoch=epoch, trainLoss=epoch_loss)

def val_one_epoch(model, criterion, dataloader, epoch, device):
    with torch.no_grad():
        model.eval()

        dataset_size = 0
        running_loss = 0

        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:
            images = data[0].to(device, dtype=torch.float)
            # this is float now I don't know it's right.
            mask_true = data[1].to(device, dtype=torch.float)

            batch_size = images.size(0)
            mask_pred = model(images)

            dice_coeff = dice_coef(mask_pred, mask_true).item()
            iou_coeff = iou_coef(mask_pred, mask_true).item()

            loss = criterion(mask_pred, mask_true) 

            running_loss += loss.item()*batch_size
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size

            bar.set_postfix(epoch=epoch, valLoss=epoch_loss, dicecoef=dice_coeff, iou_coef=iou_coeff)



if __name__ == "__main__":
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

    train_df["path"] = train_df["id"].apply(lambda x: fetch_file_from_id(train_dir, x))
    # train_dir = "./UW-Madison_GI_Tract_Image_Segmentation/input/uw-madison-gi-tract-image-segmentation/train"
    # case_id = case123_day20_slice_0001

    train_df["width"] = train_df["path"].apply(lambda x: os.path.split(x)[-1].split("_")[2]).astype("int")
    train_df["height"] = train_df["path"].apply(lambda x: os.path.split(x)[-1].split("_")[3]).astype("int")

    class_names = train_df["class"].unique()
    # --> ['large_bowel', 'small_bowel', 'stomach']
    for index, label in enumerate(class_names):
        train_df["class"].replace(label, index, inplace = True)

    # https://www.kaggle.com/code/sagnik1511/uwmgit-data-preparation-from-scratch

    # no segmentation remove
    train_df = train_df[train_df['segmentation']!='nan']
    train_df = train_df.reset_index()
    print(train_df)

    ds = UWDataset(train_df)
    print(f"Length of the dataset : {len(ds)}")

    train_size = int(len(ds)*0.8)
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(CFG.seed))

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)


    model = UNet(CFG.n_channels, CFG.n_classes).to(CFG.device)
    optimizer = optim.RMSprop(model.parameters(), lr=CFG.lr, weight_decay=1e-8, momentum=0.9)
    creterion = nn.CrossEntropyLoss()

    for epoch in range(1, CFG.epochs):
        train_one_epoch(model, optimizer, creterion, train_loader, CFG.epochs, CFG.device)
        val_one_epoch(model, creterion, val_loader, CFG.epochs, CFG.device)