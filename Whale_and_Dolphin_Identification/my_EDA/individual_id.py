import pandas as pd
import numpy as np
import os
import time
import shutil

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
ROOT_DIR = './Whale_and_Dolphin_Identification/input/happy-whale-and-dolphin'
TRAIN_DIR = './Whale_and_Dolphin_Identification/input/happy-whale-and-dolphin/train_images'
TEST_DIR = './Whale_and_Dolphin_Identification/input/happy-whale-and-dolphin/test_images'
COPY_DIR = './Whale_and_Dolphin_Identification/individual_input'



def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def get_train_file_path(id):
    return f"{TRAIN_DIR}/{id}"

createFolder(COPY_DIR)

df = pd.read_csv(f"{ROOT_DIR}/train.csv")

# 종 이름 변경
df.species.replace({"globis": "short_finned_pilot_whale",
                          "pilot_whale": "short_finned_pilot_whale",
                          "kiler_whale": "killer_whale",
                          "bottlenose_dolpin": "bottlenose_dolphin"}, inplace=True)

# file_path -> original location   copy_path -> individual_id
df['file_path'] = df['image'].apply(get_train_file_path)
df['copy_path'] = COPY_DIR+'/'+df['species']+'/'+df['individual_id']+'/'+df['image']

individual_id_array = df['individual_id'].unique()
species_array = df['species'].unique()


# # 폴더생성하려고 할때 이거 사용하기
for k in species_array:
    createFolder(f'{COPY_DIR}/{k}')

# individual_id_array에는 unique한 individual_id 배열이 들어가있음

for k in species_array:
    # species가 특정한것의 individual_id를 가져오려고 하는 task임
    df_specific_species = df[df['species']==k]

    individual_folder = df_specific_species['individual_id'].unique()
    for j in individual_folder:
        createFolder(f'{COPY_DIR}/{k}/{j}')
        print(df_specific_species['individual_id'].unique())



for i in individual_id_array:
    df_after_loc = df.loc[df['individual_id']==i]
    
    # 종 이름만 가져옴
    df_species = df_after_loc['species'].sample(1).values
    
    df_file_path = df_after_loc['file_path'].values
    df_copy_path = df_after_loc['copy_path'].values

    # print(df_species)
    # print(df_file_path)


    for j in range(len(df_file_path)):
        shutil.copyfile(df_file_path[j], df_copy_path[j])
    print(f'{df_file_path} are copied')