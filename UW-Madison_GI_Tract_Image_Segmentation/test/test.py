# import torch
# import pandas as pd

# import cv2
# import numpy as np

# # imgpath = './UW-Madison_GI_Tract_Image_Segmentation/input/uw-madison-gi-tract-image-segmentation/train/case145/case145_day19/scans/slice_0001_360_310_1.50_1.50.png'
# # iloc과 loc의 차이 ex) df.loc[], df.iloc[0] 등 integer 인덱스접근은 iloc 컬럼이름을 사용

# # path = './UW-Madison_GI_Tract_Image_Segmentation/input/uw-madison-gi-tract-image-segmentation/'

# # train = pd.read_csv(path+'train.csv')

# # train = train.dropna()
# # # print(train)
# # train = train['segmentation']
# # print(train)

# # shape = (266, 266)
# # # shape이 틀린가보다


# # segmentation = train.iloc[0]
# # segmentation = np.asarray(segmentation.split(), dtype=int)
# # # print(segmentation)


# # start_point = segmentation[0::2] - 1
# # # 왜 -1을 하는거죵? 한칸씩밀리는가요..?
# # print(f'start_point:{start_point}')

# # length_point = segmentation[1::2]
# # print(f'length_point:{length_point}')


# # end_point = start_point+length_point
# # case_mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)
# # # print(case_mask.shape)

# # for start, end in zip(start_point, end_point):
# #     case_mask[start:end] = 255


# # case_mask = case_mask.reshape((shape[0], shape[1]))

# # cv2.imshow('mask',case_mask)
# # cv2.waitKey(0)
# # print(case_mask.shape)

# # for k, location in enumerate(["large_bowel", "small_bowel", "stomach",'dd','dda']):
# #     print(k)

# # import os
# # path = '/home/User/Desktop/1_test/2_test/3_test'
# # head_tail = os.path.split(path)
# # print(head_tail)

# # final_arr = []

# # # for i in range(5):
# # #     final_arr += [1,2,51]

# # # print(final_arr)

# # k = torch.randn([5,3,2,2])
# # print(k.size())


# # print(k[4])

# # print(k[4, ...])

# # a=['a','b','c']

# # print("_".join(a))

# # k = 'case123_day20_slice_0001'

# # a = k.split('_')

# # a = ['case123', 'day20', 'slice', '0001']
# # print(a[2:])  --> ['slice', '0001']


# import keyboard

# while True:


#     keyboard.wait('k')
#     print('space was pressed! Waiting on it again...')

import torch

input = torch.randn(2,1,224,224)
target = torch.randn(2,1,224,224)

# print(input.reshape(-1).size())
inter = torch.dot(input.reshape(-1), target.reshape(-1))

print(inter)

# torch.dot()
print(inter.size())

# k = torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))
# print(k)
