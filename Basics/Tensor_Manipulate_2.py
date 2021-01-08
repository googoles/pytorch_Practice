import torch
import numpy as np


t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)

print(ft.shape)
torch.Size([2,2,3])

# 3D to 2D tensor
print(ft.view([-1,3]))
print(ft.view([-1,3]).shape)