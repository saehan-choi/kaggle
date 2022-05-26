import torch
import numpy as np
import pandas as pd
from torch import optim
from model import *
# from torch.nn.modules.activation import ReLU, Sigmoid

X_train = pd.read_csv('./input/train.csv')
X_test = pd.read_csv('./input/test.csv')
Y_train = pd.read_csv('./input/train.csv')
Y_test = pd.read_csv('./input/test.csv')

Y_train = Y_train[['Survived','Age','Embarked']]
Y_test = Y_test[['Survived','Age','Embarked']]

X_train = X_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
X_test = X_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
# print(X_train)
# 여기서 파일내부를 숫자로 바꾸어야함! neural net 예측을위해서 필수적.
X_train.dropna(subset=['Age','Embarked'], inplace=True)
X_test.dropna(subset=['Age','Embarked'], inplace=True)
Y_train.dropna(subset=['Age','Embarked'], inplace=True)
Y_test.dropna(subset=['Age','Embarked'], inplace=True)
# nan 제거작업


Y_train = Y_train['Survived']
Y_test = Y_test['Survived']

for i in range(len(X_train)):
    if X_train.iloc[i,1] == 'female':
        X_train.iloc[i,1] = 1
    elif X_train.iloc[i,1] == 'male':
        X_train.iloc[i,1] = 0

for i in range(len(X_train)):
    if X_train.iloc[i,6] == "S":
        X_train.iloc[i,6] = 1
    elif X_train.iloc[i,6] == "Q":
        X_train.iloc[i,6] = 2
    elif X_train.iloc[i,6] == "C":
        X_train.iloc[i,6] = 3

for i in range(len(X_test)):
    if X_test.iloc[i,1] == 'female':
        X_test.iloc[i,1] = 1
    elif X_test.iloc[i,1] == 'male':
        X_test.iloc[i,1] = 0

for i in range(len(X_test)):
    if X_test.iloc[i,6] == "S":
        X_test.iloc[i,6] = 1
    elif X_test.iloc[i,6] == "Q":
        X_test.iloc[i,6] = 2
    elif X_test.iloc[i,6] == "C":
        X_test.iloc[i,6] = 3



X_train = X_train.values
X_test = X_test.values
Y_train = Y_train.values
Y_test = Y_test.values

X_train = np.array(X_train,dtype=np.float32)
X_test = np.array(X_test,dtype=np.float32)
Y_train = np.array(Y_train,dtype=np.float32)
Y_test = np.array(Y_test,dtype=np.float32)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
Y_train = torch.from_numpy(Y_train)
Y_test = torch.from_numpy(Y_test)

device = torch.device('cuda')
net = NeuralNet()
net = net.to(device)


criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(),lr=1e-5)
num_epochs = 2000
batch_size = 64
batch_num_train = len(X_train) // batch_size
batch_num_test = len(X_test) // batch_size

for epochs in range(num_epochs):

    for i in range(batch_num_train):
        start = i * batch_size
        end = start + batch_size
        pred = X_train[start:end][:]
        label = Y_train[start:end]

        running_loss = 0.0
        pred, label = pred.to(device), label.to(device, dtype=torch.long)
        # pred, label = pred.unsqueeze(0), label.unsqueeze(0)
        # batch_size가 있을때는 unsqueeze를 안해도됨, 앞에 이미 텐서가 있으니
        # 그러나 텐서의 값이 하나라면 unsqueeze가 필요함
        optimizer.zero_grad()
        outputs = net(pred)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


    with torch.no_grad():
        cnt = 0
        for j in range(batch_num_test):
            start = j * batch_size
            end = start + batch_size
            pred = X_test[start:end][:]
            label = Y_test[start:end]

            acc = 0
            pred, label = pred.to(device), label.to(device, dtype=torch.long)
            # pred, label = pred.unsqueeze(0), label.unsqueeze(0)
            outputs = net(pred)                        
            _, predicted = torch.max(outputs,1)
            # label, _ = torch.max(label,0)

            for i in range(batch_size):
                if predicted[i] == label[i]:
                    cnt+=1

        print(f'accuracy:{cnt/(batch_size*batch_num_test)*100}%')

    print(f'epochs : {epochs} loss : {running_loss}')
