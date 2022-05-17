import pandas as pd
import numpy as np
import cv2


import matplotlib.pyplot as plt

from tqdm import tqdm
from glob import glob

pd.set_option('display.max_columns', 500)

path = './UW-Madison_GI_Tract_Image_Segmentation/input/uw-madison-gi-tract-image-segmentation/'
train = pd.read_csv(path+'train.csv')

def mask_from_segmentation(segmentation, shape):
    '''Returns the mask corresponding to the inputed segmentation.
    segmentation: a list of start points and lengths in this order
    max_shape: the shape to be taken by the mask
    return:: a 2D mask'''

    # Get a list of numbers from the initial segmentation
    segm = np.asarray(segmentation.split(), dtype=int)

    # Get start point and length between points
    start_point = segm[0::2] - 1
    length_point = segm[1::2]

    # Compute the location of each endpoint
    end_point = start_point + length_point

    # Create an empty list mask the size of the original image
    # take onl
    case_mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    # Change pixels from 0 to 1 that are within the segmentation
    for start, end in zip(start_point, end_point):
        case_mask[start:end] = 1

    case_mask = case_mask.reshape((shape[0], shape[1]))
    
    return case_mask

def get_image_path(base_path, df):
    '''Gets the case, day, slice_no and path of the dataset (either train or test).
    base_path: path to train image folder
    return :: modified dataframe'''
    # Create case, day and slice columns
    # id == case145_day19_slice_0049
    df["case"] = df["id"].apply(lambda x: x.split("_")[0])
    df["day"] = df["id"].apply(lambda x: x.split("_")[1])
    df["slice_no"] = df["id"].apply(lambda x: x.split("_")[-1])

    df["path"] = 0
    # segmentation이 없는 개체들 path를 0으로 만들어놓고, 있는애들만 path를 밑에서 만듬.
    n = len(df)

    # Loop through entire dataset
    for k in tqdm(range(n)):
        data = df.iloc[k, :]
        # In case coordinates for healthy tissue are present
        # pd.isnull(train.iloc[k, 2]) == train['segmentation']
        if pd.isnull(train.iloc[k, 2]) == False:
            case = data.case
            day = data.day
            slice_no = data.slice_no
            # Change value to the correct one
            df.loc[k, "path"] = glob(f"{base_path}/{case}/{case}_{day}/scans/slice_{slice_no}*")[0]
            
    return df

base_path = path+'train'
# scans. image

train = get_image_path(base_path, df=train)
# print(train.dropna().head(3))
# 지금 제생각은 segmentation 값들에 미기입된 항목이 매우많기 때문에 dropna를 해야한다는 생각인데 일단 계속해보죠!

#                            id    class  \
# 194  case123_day20_slice_0065  stomach
# 197  case123_day20_slice_0066  stomach
# 200  case123_day20_slice_0067  stomach

#                                           segmentation     case    day  \
# 194  28094 3 28358 7 28623 9 28889 9 29155 9 29421 ...  case123  day20
# 197  27561 8 27825 11 28090 13 28355 14 28620 15 28...  case123  day20
# 200  15323 4 15587 8 15852 10 16117 11 16383 12 166...  case123  day20

#     slice_no                                               path
# 194     0065  ./UW-Madison_GI_Tract_Image_Segmentation/input...
# 197     0066  ./UW-Madison_GI_Tract_Image_Segmentation/input...
# 200     0067  ./UW-Madison_GI_Tract_Image_Segmentation/input...

































# segmentation = '45601 5 45959 10 46319 12 46678 14 47037 16 47396 18 47756 18 48116 19 48477 18 48837 19 \
#                 49198 19 49558 19 49919 19 50279 20 50639 20 50999 21 51359 21 51719 22 52079 22 52440 22 52800 22 53161 21 \
#                 53523 20 53884 20 54245 19 54606 19 54967 18 55328 17 55689 16 56050 14 56412 12 56778 4 57855 7 58214 9 58573 12 \
#                 58932 14 59292 15 59651 16 60011 17 60371 17 60731 17 61091 17 61451 17 61812 15 62172 15 62532 15 62892 14 \
#                 63253 12 63613 12 63974 10 64335 7'

# shape = (310, 360)

# case_mask = mask_from_segmentation(segmentation, shape)
# wandb_mask = []
# wandb_mask.append(case_mask)


# cv2.imshow('mask',case_mask)
# cv2.waitKey(0)

# # plt.figure(figsize=(5, 5))
# # plt.title("Mask Example:")
# # plt.imshow(case_mask)
# # plt.axis("off")
# # plt.show()