import torch.nn.functional as F
import torch

img = torch.randn(1,3,224,224)
img_pad = F.pad(img, [1,2,3,4])
# >>> t4d = torch.empty(3, 3, 4, 2)
# >>> p1d = (1, 1) # pad last dim by 1 on each side
# >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
# >>> print(out.size())
# torch.Size([3, 3, 4, 4])
# >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
# >>> out = F.pad(t4d, p2d, "constant", 0)
# >>> print(out.size())
# torch.Size([3, 3, 8, 4])
# >>> t4d = torch.empty(3, 3, 4, 2)
# >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
# >>> out = F.pad(t4d, p3d, "constant", 0)
# >>> print(out.size())
# torch.Size([3, 9, 7, 3])
print(img_pad.size())