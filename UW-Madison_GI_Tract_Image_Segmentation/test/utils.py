import pandas as pd
import numpy as np
import cv2

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

from tqdm import tqdm
from glob import glob

pd.set_option('display.max_columns', 500)

path = './UW-Madison_GI_Tract_Image_Segmentation/input/uw-madison-gi-tract-image-segmentation/'
train = pd.read_csv(path+'train.csv')

my_colors = ["#CC5547", "#DB905D", "#D9AE6C", "#93AF5C", "#799042", "#61783F"]

# 그래프 그리는 겁니당
# sns.displot(
#     data=train.isna().melt(value_name="missing"),
#     #             variable  missing     melt는 이런식으로 행을 녹여서 열로변경함.
#     # 0                 id    False
#     # 1                 id    False
#     # 2                 id    False
#     # 3                 id    False
#     # 4                 id    False
#     # ...              ...      ...
#     # 346459  segmentation     True
#     # 346460  segmentation     True
#     # 346461  segmentation     True
#     # 346462  segmentation     True
#     # 346463  segmentation     True
#     y="variable",
#     hue="missing",
#     multiple="fill",
#     # Change aspect of the chart
#     aspect=3,
#     height=6,
#     # Change colors
#     palette=[my_colors[5], my_colors[2]], 
#     legend=False)

# plt.title("- [train.csv] %Perc Missing Values per variable -", size=18, weight="bold")
# plt.xlabel("Total Percentage")
# plt.ylabel("Dataframe Variable")
# plt.legend(["Missing", "Not Missing"]);
# plt.show();

# print("\n")

# # Plot 2
# plt.figure(figsize=(24,6))

# cbar_kws = { 
#     "ticks": [0, 1],
# }

# sns.heatmap(train.isna(), cmap=[my_colors[5], my_colors[2]], cbar_kws=cbar_kws)

# plt.title("- [train.csv] Missing Values per observation -", size=18, weight="bold")
# plt.xlabel("")
# plt.ylabel("Observation")
# plt.show()

class clr:
    S = '\033[1m' + '\033[92m'
    E = '\033[0m'

def CustomCmap(rgb_color):

    r1,g1,b1 = rgb_color

    cdict = {'red': ((0, r1, r1),
                   (1, r1, r1)),
           'green': ((0, g1, g1),
                    (1, g1, g1)),
           'blue': ((0, b1, b1),
                   (1, b1, b1))}

    cmap = LinearSegmentedColormap('custom_cmap', cdict)
    return cmap

mask_colors = [(1.0, 0.7, 0.1), (1.0, 0.5, 1.0), (1.0, 0.22, 0.099)]
legend_colors = [Rectangle((0,0),1,1, color=color) for color in mask_colors]
labels = ["Large Bowel", "Small Bowel", "Stomach"]

CMAP1 = CustomCmap(mask_colors[0])
CMAP2 = CustomCmap(mask_colors[1])
CMAP3 = CustomCmap(mask_colors[2])

def mask_from_segmentation(segmentation, shape):
    '''
    Returns the mask corresponding to the inputed segmentation.
    segmentation: a list of start points and lengths in this order
    max_shape: the shape to be taken by the mask
    return:: a 2D mask
    '''

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
        # train.iloc[k, 2] == train['segmentation']
        # segmentation이 있으면 실행
        if pd.isnull(train.iloc[k, 2]) == False:
            case = data.case
            day = data.day
            slice_no = data.slice_no
            # Change value to the correct one
            df.loc[k, "path"] = glob(f"{base_path}/{case}/{case}_{day}/scans/slice_{slice_no}*")[0]
            # jpg까지 되는거확인했습니당 -> glob을 이용해서 이거랑 똑같은 이미지를 좌표를 가져오게하네요
    return df

# Functions to get image width and height
def get_img_size(x, flag):
    # print(f'what is the x? : {x}')
    if x != 0:
        split = x.split("_")
        # ./UW-Madison_GI_Tract_Image_Segmentation/input/uw-madison-gi-tract-image-segmentation/train/case123/case123_day20/scans\slice_0065_266_266_1.50_1.50.png
        width = split[-4]
        height = split[-3]
    
        if flag == "width":
            return int(width)
        elif flag == "height":
            return int(height)
    return 0


def get_pixel_size(x, flag):
    
    if x != 0:
        split = x.split("_")
        # slice_0065_266_266_1.50_1.50.png
        # print(f'split:{split}')

        width = split[-2]
        height = ".".join(split[-1].split(".")[:-1])

        if flag == "width":
            return float(width)
        elif flag == "height":
            return float(height)

    return 0


base_path = path+'train'
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


# ⚕ 1.3 Image/Pixel width & height¶

# Retrieve image width and height


train["image_width"] = train["path"].apply(lambda x: get_img_size(x, "width"))
train["image_height"] = train["path"].apply(lambda x: get_img_size(x, "height"))

train["pixel_width"] = train["path"].apply(lambda x: get_pixel_size(x, "width"))
train["pixel_height"] = train["path"].apply(lambda x: get_pixel_size(x, "height"))

# print(clr.S+"train.csv now:"+clr.E)
# print(train.head(3))
#                          id        class segmentation     case    day  \
# 0  case123_day20_slice_0001  large_bowel          NaN  case123  day20
# 1  case123_day20_slice_0001  small_bowel          NaN  case123  day20
# 2  case123_day20_slice_0001      stomach          NaN  case123  day20

#   slice_no path  image_width  image_height  pixel_width  pixel_height
# 0     0001    0            0             0          0.0           0.0
# 1     0001    0            0             0          0.0           0.0
# 2     0001    0            0             0          0.0           0.0

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


def read_image(path):
    '''Reads and converts the image.
    path: the full complete path to the .png file'''

    # Read image in a corresponding manner
    # convert int16 -> float32
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')
    # Scale to [0, 255]
    image = cv2.normalize(image, None, alpha = 0, beta = 255, 
                        norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                            # 32bit float
    image = image.astype(np.uint8)
    
    return image


# def show_simple_images(sample_paths, image_names="sample_images"):
#     '''Displays simple images (without mask).'''

#     # Get additional info from the path
    
#     case_name = [info.split("_")[0][-7:] for info in sample_paths]
#     day_name = [info.split("_")[1].split("/")[0] for info in sample_paths]
#     slice_name = [info.split("_")[2] for info in sample_paths]

#     # Plot
#     fig, axs = plt.subplots(2, 5, figsize=(23, 8))
#     axs = axs.flatten()

#     for k, path in enumerate(sample_paths):
#         title = f"{k+1}. {case_name[k]} - {day_name[k]} - {slice_name[k]}"
#         axs[k].set_title(title, fontsize = 14, 
#                          color = my_colors[-1], weight='bold')

#         img = read_image(path)
#         axs[k].imshow(img)
#         axs[k].axis("off")

#     plt.tight_layout()
#     plt.show()
#     # 🐝 Log Image to W&B


# CASE = "case123"

# # Sample a few images from speciffied case
# sample_paths1 = train[(train["segmentation"].isna()==False) & (train["case"]==CASE)]["path"]\
#                 .reset_index().groupby("path")["index"].count()\
#                 .reset_index().loc[:9, "path"].tolist()

# show_simple_images(sample_paths1, image_names="case123_samples")


def get_id_mask(ID, verbose=False):
    '''
    Returns a mask for each case ID. If no segmentation was found, the mask will be empty
    - meaning formed by only 0
    ID: the case ID from the train.csv file
    verbose: True if we want any prints
    return: segmentation mask
    '''
    # ~~~ Get the data ~~~
    # Get the portion of dataframe where we have ONLY the speciffied ID
    
    ID_data = train[train["id"]==ID].reset_index(drop=True)
    # 여기서는 들어온 ID만 판별하기 때문에 들어온 이미지에 대해서만 결과를 도출합니다.


    # Split the dataframe into 3 series of observations
    # each for one speciffic class - "large_bowel", "small_bowel", "stomach"
    observations = [ID_data.loc[k, :] for k in range(3)]

    # what is observation? : [id                                        case131_day0_slice_0066
    # class                                                 large_bowel
    # segmentation    45601 5 45959 10 46319 12 46678 14 47037 16 47...
    # case                                                      case131
    # day                                                          day0
    # slice_no                                                     0066
    # path            ./UW-Madison_GI_Tract_Image_Segmentation/input...
    # image_width                                                   360
    # image_height                                                  310
    # pixel_width                                                   1.5
    # pixel_height                                                  1.5
    # Name: 0, dtype: object, id                                        case131_day0_slice_0066
    # class                                                 small_bowel
    # segmentation    54957 4 55301 9 55315 8 55660 24 56019 27 5637...
    # case                                                      case131
    # day                                                          day0
    # slice_no                                                     0066
    # path            ./UW-Madison_GI_Tract_Image_Segmentation/input...
    # image_width                                                   360
    # image_height                                                  310
    # pixel_width                                                   1.5
    # pixel_height                                                  1.5
    # Name: 1, dtype: object, id                                        case131_day0_slice_0066
    # class                                                     stomach
    # segmentation    43410 9 43763 18 44121 21 44480 23 44840 23 45...
    # case                                                      case131
    # day                                                          day0
    # slice_no                                                     0066
    # path            ./UW-Madison_GI_Tract_Image_Segmentation/input...
    # image_width                                                   360
    # image_height                                                  310
    # pixel_width                                                   1.5
    # pixel_height                                                  1.5
    # Name: 2, dtype: object]

    # ~~~ Create the mask ~~~
    # Get the maximum height out of all observations
    # if max == 0 then no class has a segmentation
    # otherwise we keep the length of the mask
    max_height = np.max([obs.image_height for obs in observations])
    max_width = np.max([obs.image_width for obs in observations])

    # Get shape of the image
    # 3 channels of color/classes
    shape = (max_height, max_width, 3)

    # Create an empty mask with the shape of the image
    mask = np.zeros(shape, dtype=np.uint8)

    # If there is at least 1 segmentation found in the group of 3 classes
    if max_height != 0:
        for k, location in enumerate(["large_bowel", "small_bowel", "stomach"]):
            observation = observations[k]
            segmentation = observation.segmentation

            # If a segmentation is found
            # Append a new channel to the mask
            if pd.isnull(segmentation) == False:
                mask[..., k] = mask_from_segmentation(segmentation, shape)
                # zeros에서 0채널을 large_bowel, 1채널을 small_bowel, 2채널을 stomach로 해놓네요 ㅎ
                # mask_from_segmentation에는 채널없이 2d image만 반환합니다 ㅎ 그걸 저기서 채널을 이용해서 받아들이네요

    # If no segmentation was found skip
    # elif max_segmentation == 0: 여기 오타같네요 segmentation으로 변경합니다.
    elif segmentation == 0:
        mask = None
        if verbose:
            print("None of the classes have segmentation.")

    return mask

# Full Example

# Read image
path = base_path + '/case131/case131_day0/scans/slice_0066_360_310_1.50_1.50.png'
img = read_image(path)

# Get mask
ID = "case131_day0_slice_0066"
mask = get_id_mask(ID, verbose=False)



def plot_original_mask(img, mask, alpha=1):
    # Change pixels - when 1 make True, when 0 make NA
    mask = np.ma.masked_where(mask == 0, mask)
    # array([0, 1, 2])  -->  mask == 0 -->  data=[--, 1, 2] 이런식으로 진행됨.

    # Split the channels
    mask_largeB = mask[:, :, 0]
    mask_smallB = mask[:, :, 1]
    mask_stomach = mask[:, :, 2]


    # Plot the 2 images (Original and with Mask)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    # Original
    ax1.set_title("Original Image")
    ax1.imshow(img)
    ax1.axis("off")

    # With Mask
    ax2.set_title("Image with Mask")
    ax2.imshow(img)
    ax2.imshow(mask_largeB, interpolation='none', cmap=CMAP1, alpha=alpha)
    ax2.imshow(mask_smallB, interpolation='none', cmap=CMAP2, alpha=alpha)
    ax2.imshow(mask_stomach, interpolation='none', cmap=CMAP3, alpha=alpha)
    ax2.legend(legend_colors, labels)
    ax2.axis("off")
    
#     fig.savefig('foo.png', dpi=500)
    plt.show()

plot_original_mask(img, mask, alpha=1)



# plot이 안되서
# plot_original_mask(img, mask, alpha=1)

# https://www.kaggle.com/code/andradaolteanu/aw-madison-eda-in-depth-mask-exploration/notebook