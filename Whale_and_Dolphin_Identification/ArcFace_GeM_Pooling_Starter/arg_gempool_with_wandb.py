import os
import gc
import cv2
import math
import copy
import time
import random
from kaggle_secret import secret_key

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# For Image Models
import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# WANDB 켤거면 이거 해제하세요 ㅎ 다른건 다 똑같슴다
import wandb
try:
    api_key = secret_key()
    wandb.login(key=api_key)
    anony = None
except:
    anony = "must"
    print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

CONFIG = {"seed": 2022,
          "epochs": 4,
          "img_size": 448,
          "model_name": "tf_efficientnet_b0_ns",
          "num_classes": 15587,
          "embedding_size": 512,
          "train_batch_size": 8,
          # batch_size 수정함
          "valid_batch_size": 16,
          # gpu사용량보면 validation set 실행시 별로 사용을 안함을 알수이씀
          "learning_rate": 1e-4,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-6,
          "T_max": 500,
          "weight_decay": 1e-6,
          "n_fold": 5,
          "n_accumulate": 1,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          # ArcFace Hyperparameters
          "s": 30.0, 
          "m": 0.50,
          "ls_eps": 0.0,
          "easy_margin": False
          }

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

class HappyWhaleDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['individual_id'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]
        
        # 라벨은 individual_id

        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long)
        }

# print(df['individual_id'].max())
# 15586 개의 individual_id 존재

data_transforms = {
    "train": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.ShiftScaleRotate(shift_limit=0.1, 
                           scale_limit=0.15, 
                           rotate_limit=60, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),
    
    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        # albumentations 에서는 normalize 후 tensor로 변환
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
}



class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        # torch.ones -> tensor([1.])
        # nn.Parameter(torch.ones(1)*p) -> tensor([3.], requires_grad=True)   if p==3:
        # 학습가능한 상태로 변경됨
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        # avg_pool2d(input, kernel_size)
        # avg pool에서 지수부분만 2곳 바뀜  (pow를 통해)

    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'



# 이부분 재 이해가 필요
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, 
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        # in_feature -> embedding_size   config에 512로 되어있음 ㅎ
        # out_feature -> num_class    config에 15587로 되어있음 -> indivisual_id 갯수임
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # F.linear 하면 앞뒤쪽이 transform 됨
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        # math.pi = 3.141592 ....
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        # margin이 0.5이므로 각도 30도와 비슷 정확히는 28.6도에 가까움
        # 3.14 -> pi 와 같음

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # input [8, 512] 이렇게 들어옴 input = [batch_size, embedding]
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # sine = root(1-cos^2)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
            # easy margin 으로 한다고 가정하자 밑의 식은 이해가 잘 안됨. 

        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=CONFIG['device'])
        # one hot_zeros shape:torch.Size([8, 15587])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # one hot_after_scatter shape:torch.Size([8, 15587])
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        

        # print(f'label : {label}')
        # label : tensor([12927, 11406,  3251, 15569, 14304, 11598, 12602,  9524], device='cuda:0')        
        # print(f'label.view : {label.view(-1,1)}')
        # label.view : tensor([[12927],
        #         [11406],
        #         [ 3251],
        #         [15569],
        #         [14304],
        #         [11598],
        #         [12602],
        #         [ 9524]], device='cuda:0')   label을 onehot에다가 정렬해서 넣는듯

        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

class HappyWhaleModel(nn.Module):
    def __init__(self, model_name, embedding_size, pretrained=True):
        super(HappyWhaleModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        # 만약에 global_pool 을 빼면 torch.size([8, 1280]) 을 반환  빼지않으면 torch.size([8, 1280, 14, 14])
        # infeatures는 기존 모델의 features 를 저장하는것
        # classifier, global_pool은 기존의 tf_efficientnet_b0_ns 모델의 구성을 identity로 변경
        # -> 내가 나중에 직접 조정해줘야하는듯 ㅎ  test 에가면 실험한거 볼 수 있습니다.
        self.pooling = GeM()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.fc = ArcMarginProduct(embedding_size, 
                                   CONFIG["num_classes"],
                                   s=CONFIG["s"], 
                                   m=CONFIG["m"], 
                                   easy_margin=CONFIG["ls_eps"], 
                                   ls_eps=CONFIG["ls_eps"])


    def forward(self, images, labels):


        features = self.model(images)
              
        # print(f'features shape:{features.shape}')   ([8, 1280, 14, 14])     ([batch_size, 1280, 14, 14])
        pooled_features = self.pooling(features).flatten(1)

        # print(f'pool shape:{self.pooling(features).shape}')   torch.Size([8, 1280, 1, 1])
        # 아 이게 kernel_size가 image size라서 ([8, 1280, 1, 1]) 이런식으로 output이 나오네요!

        # print(f'pool      :{self.pooling(features).flatten(1).shape}')  torch.Size([8, 1280])
        # print(f'pool      :{self.pooling(features).flatten(2).shape}')  torch.Size([8, 1280, 1])

        # feature는 모델을 거친결과가 아님!!!  flatten() = flatten(0)   flatten(1) = 배열의 첫번째에서 flatten 하는것과 같음.
        # 이부분 솔직히 이해못했음

        embedding = self.embedding(pooled_features)
        # print(embedding.shape)  torch.Size([8, 512])       nn.linear(1280, 512) 인듯
        output = self.fc(embedding, labels)
        return output

    def extract(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        # 여기서 pooled_features는 GeMpooling 이므로,
        embedding = self.embedding(pooled_features)
        return embedding


def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)
    
        batch_size = images.size(0)
        # 첫번째 배치사이즈를 가져옴
    
        outputs = model(images, labels)
        # print(f'outputs:{outputs}\n')
        # print(f'labels:{labels}')
        loss = criterion(outputs, labels)

        # 만약 n_accumulate 가 2라면?
        # loss를 2로 나누고 backpropagation을 진행
        loss = loss / CONFIG['n_accumulate']
            
        loss.backward()
        # 변화도가 누적된 grad tensor를 가지게됨.


        # 첫번째는 가중치를 업데이트 하지않는다.
        # 두번째때 가중치를 업데이트..  accumulate가 2라면 ㅎ
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss

@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)
        
        batch_size = images.size(0)

        outputs = model(images, labels)
        loss = criterion(outputs, labels)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)
        # bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
        #                 LR=optimizer.param_groups[0]['lr'])
        # validation에는 optimizer가 있을필요가 없어서 제거했음 에러나면 알아서 할것.
    gc.collect()
    return epoch_loss


def run_training(model, optimizer, scheduler, device, num_epochs):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], epoch=epoch)
        
        val_epoch_loss = valid_one_epoch(model, valid_loader, device=CONFIG['device'], 
                                         epoch=epoch)
        
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        
        # Log the metrics
        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})
        
        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            # run.summary["Best Loss"] = best_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "Loss{:.4f}_epoch{:.0f}.pt".format(best_epoch_loss, epoch)
            # pt로 바꿨음 bin 말고 안익숙해스ㅎ,,
            # 오류나면 다시 pt로 바꾸기
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler


def prepare_loaders(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = HappyWhaleDataset(df_train, transforms=data_transforms["train"])
    valid_dataset = HappyWhaleDataset(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=2, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    ROOT_DIR = './Whale_and_Dolphin_Identification/input/happy-whale-and-dolphin'
    TRAIN_DIR = './Whale_and_Dolphin_Identification/input/happy-whale-and-dolphin/train_images'
    TEST_DIR = './Whale_and_Dolphin_Identification/input/happy-whale-and-dolphin/test_images'
    
    set_seed(CONFIG['seed'])

    def get_train_file_path(id):
        return f"{TRAIN_DIR}/{id}"

    df = pd.read_csv(f"{ROOT_DIR}/train.csv")
    
    df['file_path'] = df['image'].apply(get_train_file_path)

    # 라벨을 문자 -> 숫자별로 나눔
    encoder = LabelEncoder()
    df['individual_id'] = encoder.fit_transform(df['individual_id'])

    # pkl 왜 사용하는지 모르겠어서 없애놓을게여
    # with open("le.pkl", "wb") as fp:
    #     joblib.dump(encoder, fp)

    # 레이블별로 뭉치는걸 방지하기위해 individual_id를 기준으로 StratifiedKFold를 사용
    skf = StratifiedKFold(n_splits=CONFIG['n_fold'])

    for fold, ( _, val_) in enumerate(skf.split(X=df, y=df.individual_id)):
        df.loc[val_ , "kfold"] = fold

    model = HappyWhaleModel(CONFIG['model_name'], CONFIG['embedding_size'])
    model.to(CONFIG['device'])

    train_loader, valid_loader = prepare_loaders(df, fold=0)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                        weight_decay=CONFIG['weight_decay'])
    scheduler = fetch_scheduler(optimizer)

    run = wandb.init(project='HappyWhale', 
                     config=CONFIG,
                     job_type='Train',
                     tags=['arcface', 'gem-pooling', 'effnet-b0-ns', '448'],
                     anonymous='must')


    model, history = run_training(model, optimizer, scheduler,
                                device=CONFIG['device'],
                                num_epochs=CONFIG['epochs'])

    run.finish()
