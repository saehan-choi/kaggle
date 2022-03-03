import os
from glob import glob
from tqdm.notebook import tqdm
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import timm

import imagesize

try:
    from cuml import TSNE # if gpu is ON
except:
    from sklearn.manifold import TSNE # for cpu
    
import wandb
import IPython.display as ipd

class CFG:
    seed          = 42
    base_path     = './input/happy-whale-and-dolphin'
    embed_path    = './input/happywhale-embedding-dataset'
    num_samples   = None #  None for all samples
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    competition   = 'happywhale'
    _wandb_kernel = 'awsaf49'


def seed_torch(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('# SEEDING DONE')
seed_torch(CFG.seed)

# if you wanna use wandb, use it
# try:
#     user_secrets = '72822190acbfe6e32f48e9c652274c39aea9187b'
#     api_key = user_secrets
#     wandb.login(key=api_key)
#     anonymous = None
#     print('성공함')

# except:
#     anonymous = "must"
#     wandb.login(anonymous=anonymous)
#     print('To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')



df = pd.read_csv(f'{CFG.base_path}/train.csv')
df['image_path'] = CFG.base_path+'/train_images/'+df['image']
df['split'] = 'Train'

test_df = pd.read_csv(f'{CFG.base_path}/sample_submission.csv')
test_df['image_path'] = CFG.base_path+'/test_images/'+test_df['image']
test_df['split'] = 'Test'

# print('Train Images: {:,} | Test Images: {:,}'.format(len(df), len(test_df)))


# convert beluga, globis to whales
df.loc[df.species.str.contains('beluga'), 'species'] = 'beluga_whale'
df.loc[df.species.str.contains('globis'), 'species'] = 'globis_whale'

df['class'] = df.species.map(lambda x: 'whale' if 'whale' in x else 'dolphin')
# 문자형 값들이 map함수를 거치면 숫자형으로 변환됨
# ex) A, B, C, C, B -> 1, 2, 3, 3, 2


# fix duplicate(복제하다) labels
# https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/304633
df['species'] = df['species'].str.replace('bottlenose_dolpin','bottlenose_dolphin')
df['species'] = df['species'].str.replace('kiler_whale','killer_whale')

# Find Image Size
def get_imgsize(row):
    row['width'], row['height'] = imagesize.get(row['image_path'])
    return row

# Train
tqdm.pandas(desc='Train ')
df = df.progress_apply(get_imgsize, axis=1)
print('Train:')
ipd.display(df.head(2))

# Test
tqdm.pandas(desc='Test ')
test_df = test_df.progress_apply(get_imgsize, axis=1)
print('Test:')
ipd.display(test_df.head(2))



if CFG.num_samples:
    df = df.iloc[:CFG.num_samples]
    test_df = test_df.iloc[:CFG.num_samples]


data = df.species.value_counts().reset_index()
# print(data) 이거하면 각 인덱스별로 몇개있는지 나옴
fig = px.bar(data, x='index', y='species', color='species',title='Species', text_auto=True)
# color 없애면 똑같은 색갈로나오네
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
# fig.show()
# bar chart를 만들어냄



data = df['class'].value_counts().reset_index()

fig = px.bar(data, x='index', y='class', color='class', title='Whale Vs Dolphin', text_auto=True)
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
# fig.show()
# https://www.kaggle.com/awsaf49/happywhale-data-distribution

# print(df)


# width, height 로 EDA를 할 수 있음 class vs iamgesize
# 이미지의 분포가 어디서부터 어디까지 이뤄지는지
# It is visible that Distribution of ImageSize is similar for both Whale and Dolphin except some cases in height.

fig = px.histogram(df,
                   x="width", 
                   color="class",
                   barmode='group',
                   log_y=True,
                   title='Width Vs Class')
# fig.show()

fig = px.histogram(df,
                   x="height", 
                   color="class",   
                   barmode='group',
                   log_y=True,
                   title='Height Vs Class')
# fig.show()


# ImageSize Vs Split(Train/Test)
# It can be notices that distribution of width for train and test data, looks quite similar. So, we can resize without any tension.
# For height we have some unique shapes.

fig = px.histogram(pd.concat([df, test_df]),
                   x="width", 
                   color="split",
                   barmode='group',
                   log_y=True,
                   title='Width Vs Split')
# fig.show()

fig = px.histogram(pd.concat([df, test_df]),
                   x="height", 
                   color="split",
                   barmode='group',
                   log_y=True,
                   title='Height Vs Split')
# fig.show()


# Data Pipeline 🍚
# To create image embedding we will,  (embedding -> 고차원을 저차원으로 변환(from high dimension to low dimension))
# Read the image.
# Resize it accordingly.

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # opencv 가 BGR로 이루어지기 때문에 RGB 채널로 변환
    return img


class ImageDataset(Dataset):
    def __init__(self,
                 path,
                 target=None,
                 input_shape=(128, 256),
                #  input_shape에 큰 의미없음 나중에 224, 224로 받아옴
                 transform=None,
                 channel_first=True,
                ):
        super(ImageDataset, self).__init__()
        # super().__init__()
        # super()로 기반 클래스(부모 클래스)를 초기화해줌으로써, 기반 클래스의 속성을 subclass가 받아오도록 한다. 
        # (초기화를 하지 않으면, 부모 클래스의 속성을 사용할 수 없음)

        self.path = path
        self.target = target
        self.input_shape = input_shape
        self.transform = transform
        self.channel_first = channel_first

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        img = load_image(self.path[idx])
        img = cv2.resize(img, dsize=self.input_shape)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        if self.channel_first:
            img = img.transpose((2, 0, 1))
        if self.target is not None:
            target = self.target[idx]
            return img, target
        else:
            return img

def get_dataset(path, target=None, batch_size=32, input_shape=(224, 224)):

    # 나중에 batch_size도 config로 고칠수 있겠다.
    dataset = ImageDataset(path=path,
                           target=target,
                           input_shape=input_shape,
                          )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        # num_worker가 2로 되있었는데, 0으로 고치면 dataloader에서 error 안남
        # num_worker는 gpu를 효율적으로 쓰기위해 있는걸로암. 프로젝트 다 끝나면 num_worker = 2 로 해볼것
        shuffle=False,
        pin_memory=True,
    )
    return dataloader


# 이건아직 안봤음 visualization 할 때 필요
def plot_batch(batch, row=2, col=2, channel_first=True):
    if isinstance(batch, tuple) or isinstance(batch, list):
        imgs, tars = batch
    else:
        imgs, tars = batch, None
    plt.figure(figsize=(col*3, row*3))
    for i in range(row*col):
        plt.subplot(row, col, i+1)
        img = imgs[i].numpy()
        if channel_first:
            img = img.transpose((1, 2, 0))
        plt.imshow(img)
        if tars is not None:
            plt.title(tars[i])
        plt.axis('off')
    plt.tight_layout()
    # plt.show()
    

def gen_colors(n=10):
    cmap   = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, n + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    return colors



if __name__ == '__main__':
    
    train_loader = get_dataset(path=df.image_path.tolist(),
                        target=df.species.tolist(),
                        input_shape=(224,224),
                        )
                        
    test_loader = get_dataset(path=test_df.image_path.tolist(),
                        target=None,
                        input_shape=(224,224),
                        )


    batch = iter(train_loader).next()
    plot_batch(batch, row=2, col=5)

    batch = iter(test_loader).next()
    plot_batch(batch, row=2, col=5)


    class ImageModel(nn.Module):
        def __init__(self, backbone_name, pretrained=True):
            super(ImageModel, self).__init__()
            self.backbone = timm.create_model(backbone_name,
                                            pretrained=pretrained)
            self.backbone.reset_classifier(0) # to get pooled features
            #   classification에 관련된 layer들 없앰 
            #   했을때 -> 안했을때
            #   (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
            #   (classifier): Linear(in_features=1280, out_features=1000, bias=True)

            #   (global_pool): SelectAdaptivePool2d (pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
            #   (classifier): Identity()
            #   여기서 output으로 1280개의 벡터가 생성됨.

        def forward(self, x):            
            x = self.backbone(x)
            return x

    model = ImageModel('tf_efficientnet_b0')

    @torch.no_grad()
    # @ python에 대한 설명은 https://choice-life.tistory.com/42 여기 잘 나와있습니다.

    def predict(model, dataloader):
        model.eval() # turn off layers such as BatchNorm or Dropout
        model.to(CFG.device) # cpu -> gpu
        embeds = []
        pbar = tqdm(dataloader, total=len(dataloader))
        for img in pbar:
            img = img.type(torch.float32) # uint8 -> float32
            img = img.to(CFG.device) # cpu -> gpu
            embed = model(img) # this is where magic happens ;)
            gpu_mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            pbar.set_postfix(gpu_mem=f'{gpu_mem:0.2f} GB')
            embeds.append(embed.cpu().detach().numpy())
        return np.concatenate(embeds)

    @torch.no_grad()
    def predict(model, dataloader):
        model.eval() # turn off layers such as BatchNorm or Dropout
        # 이게 없으면 batchnorm이랑 dropout 안꺼진 상태에서 진행됨
        model.to(CFG.device) # cpu -> gpu
        embeds = []
        pbar = tqdm(dataloader, total=len(dataloader))
        for img in pbar:
            img = img.type(torch.float32) # uint8 -> float32
            img = img.to(CFG.device) # cpu -> gpu
            embed = model(img) # this is where magic happens ;)
            gpu_mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            pbar.set_postfix(gpu_mem=f'{gpu_mem:0.2f} GB')
            # 이거 주석처리했을때랑 주석처리 안했을때랑 차이보기..
            # set_postfix가 뭔지 잘 모르겠네요.
            embeds.append(embed.cpu().detach().numpy())
        return np.concatenate(embeds)


    train_loader = get_dataset(
        path=df.image_path.tolist(),
        target=None,
        input_shape=(224,224),
        batch_size=128,
        # 128*4 에서 128로줄였음
    )

    test_loader = get_dataset(
        path=test_df.image_path.tolist(),
        target=None,
        input_shape=(224,224),
        batch_size=128*3,
        # 128*4 에서 128로줄였음
    )

    # if CFG.embed_path:
    #     print('# Train Embeddings:')
    #     train_embeds = np.load(f'{CFG.embed_path}/train_embeds.npy')
        
    #     print('# Test Embeddings:')
    #     test_embeds = np.load(f'{CFG.embed_path}/test_embeds.npy')
        
    # else:
    print('# Train Embeddings:')
    train_embeds = predict(model, train_loader)
    np.save(f'{CFG.embed_path}/train_embeds.npy', train_embeds) # save embeddings for reuse

    print('# Test Embeddings:')
    test_embeds = predict(model, test_loader)
    np.save(f'{CFG.embed_path}/train_embeds.npy', test_embeds) # save embeddings for reuse


    tsne = TSNE()

    # Concatenate both train and test
    embeds = np.concatenate([train_embeds,test_embeds])

    # Fit TSNE on the embeddings and then transfer data
    embed2D = tsne.fit_transform(embeds)

    print(embed2D)
    print(embed2D.shape)

    # Train
    df['x'] = embed2D[:len(train_embeds),0]
    df['y'] = embed2D[:len(train_embeds),1]

    # Test
    test_df['x'] = embed2D[len(train_embeds):,0]
    test_df['y'] = embed2D[len(train_embeds):,1]


    # convert config from class to dict
    config = {k:v for k,v in dict(vars(CFG)).items() if '__' not in k}
    # config파일 __main__, __getitem__ 같은 필요없는것들은 제거하고 dictionary형태로 저장함
    # vars는 관련된 모든(?) 정보를 뱉어냄 ㅎ

    # initialize wandb project
    wandb.init(project='happywhale-public', config=config)

    # process data for wandb
    wdf1 = pd.concat([df, test_df]).drop(columns=['image_path','predictions']) # train + test
    wdf2 = df.copy() # only train as some columns of test don't have any value e.g: species

    # log the data
    wandb.log({"All":wdf1, 
            "Train":wdf2}) # log both result

    # save embeddings to wandb for later use
    wandb.save('test_embeds.npy'); # save train embeddings
    wandb.save('train_embeds.npy'); # save test embeddings

    # show wandb dashboard
    ipd.display(ipd.IFrame(wandb.run.url, width=1080, height=720)) # show wandb dashboard

    # finish logging
    wandb.finish()
    

    