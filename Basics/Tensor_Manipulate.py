import numpy as np
import torch


# t = np.array([0,1,2,3,4,5,6])
# print(t)
# print("Rank of t: ", t.ndim)
# print("Shape of t: ", t.shape)

t = torch.FloatTensor([0, 1, 2, 3, 4, 5, 6])
print(t)

print(t.dim())
print(t.shape)
print(t.size())

# Tensor Slicing

print(t[0], t[1], t[-1])  # 인덱스로 접근
print(t[2:5], t[4:-1])    # 슬라이싱
print(t[:2], t[3:])       # 슬라이싱


# 2D with Pytorch

t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])
print(t)
print(t.dim())  # rank. 즉, 차원
print(t.size()) # shape
print(t[:, 1]) # 첫번째 차원을 전체 선택한 상황에서 두번째 차원의 첫번째 것만 가져온다.
print(t[:, 1].size()) # ↑ 위의 경우의 크기
