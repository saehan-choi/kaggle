from model import *
import torch
import pandas as pd
from utils import *

path = './weights/'
sample_sub = "./input/test.csv"
test_data = pd.read_csv(sample_sub) 
pred = []


net = NeuralNet()
net.load_state_dict(torch.load(path+'model_alot_of_feature.pt'))
net.eval()


bins = [-0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
        2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

test_data['time_step'] = pd.cut(x=test_data['time_step'], bins=[-0.99, 0.5, 1, 1.5, 2, 2.5, 3], labels=[0,1,2,3,4,5])
test_data['time_step'] = test_data['time_step'].astype('int64')

test_data = pd.concat([test_data, change_columns('C', test_data)],axis=1)
test_data = pd.concat([test_data, change_columns('R', test_data)],axis=1)


test_data = create_new_feat(test_data)
test_data = test_data.fillna(test_data.min())

test_data = test_data.drop(['id','breath_id'], axis=1)

print(test_data)

test_data = numpy_to_tensor(test_data)
test_data = net(test_data)
pred = test_data.tolist()
# tensor를 배열로 바꿔줌 엑셀에다 값을 넣기위해

empty = []
for i in pred:
    empty.append(i[0])

pred[:]=empty[:]


sample = pd.read_csv('./input/sample_submission.csv')
# sample['id'] = test_data['id']
sample['pressure'] = pd.DataFrame({'pressure': pred[:]})
sample.head()
sample.to_csv('submission_quantile.csv', index=False)