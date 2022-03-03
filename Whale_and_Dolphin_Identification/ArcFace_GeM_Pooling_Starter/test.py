import torch

import torch.nn as nn

import math

import torch.nn.functional as F

# print(nn.Parameter(torch.ones(1)*3).shape)

# print(torch.ones(1))

# print(nn.Parameter(torch.FloatTensor(4,5)))

# print(math.cos(0.5))

# print(math.sin(0.5))

# a = nn.Parameter(torch.FloatTensor(4))

# b = nn.Parameter(torch.FloatTensor(4))

# cosine = F.linear(a,4)

# print(cosine)

# cosine = torch.randn(3,2)
# print(cosine)
# phi = torch.ones(3,2)

# print(phi)

# phi = torch.where(cosine > 0, phi, cosine)



# print(phi)


# out_features = 4
# in_features = 4

# weight = nn.Parameter(torch.randn(25, 512))


# print(weight)

# inp = nn.Parameter(torch.randn(512))
# # 이게 자동으로 transfer 하네 linear할떄
# print(inp)

# res = F.linear(inp,weight)


# print(res)
# print(res.shape)

# t = torch.tensor([[[1, 2, 4],
#                     [3, 4, 2]],
#                     [[5, 6, 6],
#                     [7, 8, 2]]])

# print(t.shape)

# print(t.size(2))

# import timm

# model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
# print(f'model:{model} \n this is a model')
# in_features = model.classifier.in_features

# print(f'this is the in_features:{in_features}')

# model.classifier = nn.Identity()
# model.global_pool = nn.Identity()

# print(f'model:{model} \n this is a model')


# print(nn.Linear(1024, 2))

import pandas as pd

pd.set_option('display.max_columns', None)

ROOT_DIR = './Whale_and_Dolphin_Identification/input/happy-whale-and-dolphin'
TRAIN_DIR = './Whale_and_Dolphin_Identification/input/happy-whale-and-dolphin/train_images'
TEST_DIR = './Whale_and_Dolphin_Identification/input/happy-whale-and-dolphin/test_images'


def get_train_file_path(id):
    return f"{TRAIN_DIR}/{id}"

df = pd.read_csv(f"{ROOT_DIR}/train.csv")
print(df['species'])



# 같은 individual_id를 가지는 애들끼리 묶어보자