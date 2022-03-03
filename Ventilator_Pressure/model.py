import torch.nn as nn
from torch.nn.modules.container import Sequential

class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(51, 512)
        self.layer1 = self.make_layers(512, num_repeat=4)
        self.fc5 = nn.Linear(512, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = nn.Dropout(0.2)(x)
        x = self.fc5(x)
        return x

    def make_layers(self, value, num_repeat):
        layers = []
        for _ in range(num_repeat):
            layers.append(nn.Linear(value, value))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)



# class NeuralNet(nn.Module):

#     def __init__(self):
#         super(NeuralNet, self).__init__()
#         self.layer1 = self.make_layers(512, 4)
        
#     def forward(self, x):
#         x = self.layer1(x)

#         return x

#         epochs : 0

#     def make_layers(self, value, num_repeat):
#         layers = []
#         layers.append(nn.Linear(6,512))
#         layers.append(nn.ReLU(inplace=True))
#         for i in range(num_repeat):
#             layers.append(nn.Linear(value, value))
#             layers.append(nn.ReLU(inplace=True))

#         layers.append(nn.Linear(512,1))
#         return nn.Sequential(*layers)