import torch
import torchvision
from torchvision import transforms

import cv2

from efficientnet_pytorch import EfficientNet
import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


img = cv2.imread('./input/happy-whale-and-dolphin/train_images/0a0cedc8ac6499.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)

# tensor file
tf = transforms.ToTensor()
img = tf(img)
img = img.permute(2,1,0)
# tensor change

efficient_model = EfficientNet.from_pretrained('efficientnet-b7')
image_effnet = efficient_model(img).to(device)
print(image_effnet)


