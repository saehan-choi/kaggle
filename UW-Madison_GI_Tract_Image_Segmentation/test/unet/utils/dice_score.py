import torch
from torch import Tensor

def dice_coef(y_pred, y_true, thr=0.5, dim=(2,3), epsilon=0.001):
    assert y_true.size() == y_pred.size()
    y_true = y_true.to(torch.float32)
    
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    
    print(f'inter:{inter}')
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    print(f'den:{den}')
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    print(f'dice:{dice}')
    return dice


def iou_coef(y_pred, y_true, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou


# def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
#     # Average of Dice coefficient for all batches, or for a single mask
#     assert input.size() == target.size()
#     if input.dim() == 2 and reduce_batch_first:
#         raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

#     if input.dim() == 2 or reduce_batch_first:
#         inter = torch.dot(input.reshape(-1), target.reshape(-1))
        
#         sets_sum = torch.sum(input) + torch.sum(target)

#         if sets_sum.item() == 0:
#             sets_sum = 2 * inter


#         return (2 * inter + epsilon) / (sets_sum + epsilon)
#     else:
#         # compute and average metric for each batch element
#         dice = 0
#         for i in range(input.shape[0]):
#             dice += dice_coeff(input[i, ...], target[i, ...])
#         return dice / input.shape[0]


# def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
#     # Average of Dice coefficient for all classes
#     assert input.size() == target.size()
#     dice = 0
#     for channel in range(input.shape[1]):
#         dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

#     return dice / input.shape[1]


# def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
#     # Dice loss (objective to minimize) between 0 and 1
#     assert input.size() == target.size()
#     fn = multiclass_dice_coeff if multiclass else dice_coeff
#     return 1 - fn(input, target, reduce_batch_first=True)