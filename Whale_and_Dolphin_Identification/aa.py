import torch
import torch.nn as nn

import timm


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

    def forward(self, x):            
        x = self.backbone(x)
        return x

model = ImageModel('tf_efficientnet_b0')

print(model)