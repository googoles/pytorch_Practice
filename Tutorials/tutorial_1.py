
from __future__ import print_function
import torch
import numpy as np

x = torch.empty(5, 3)  # empty matrix
print(x)

a = torch.rand(5, 3)  # random matrix Dtype = long
print(a)
b = torch.zeros(5, 3, dtype=torch.long)
# Dtype이 long이고 0으로 채워진 행렬을 생성
print(b)
x = torch.tensor([5.5, 3])
# Tensor 직접 생성
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)

print(x.size())  # size of matrix

# Operations

y = torch.rand(5, 3)
print(x + y)  # Add 1
print(torch.add(x, y))  # Add 2

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# Result tensor as a parameter

y.add_(x)
print(y)
# indexing
print(x[:, 1])
# Size Change of Tensor
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # Guess -1 from another dimension
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())
# Get item's value


# Transformation Torch Tensor to Numpy
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# Transformation Numpy to Torch Tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)


# Cuda Tensors

# 이 코드는 CUDA가 사용 가능한 환경에서만 실행합니다.
# ``torch.device`` 를 사용하여 tensor를 GPU 안팎으로 이동해보겠습니다.
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # CUDA 장치 객체(device object)로
#     y = torch.ones_like(x, device=device)  # GPU 상에 직접적으로 tensor를 생성하거나
#     x = x.to(device)                       # ``.to("cuda")`` 를 사용하면 됩니다.
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))       # ``.to`` 는 dtype도 함께 변경합니다!


if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x,device=device)
    x = x.to(device)
    z = x+y
    print(z)
    print(z.to("cpu", torch.double))

