# 숫자이미지 분류

'''
신경망의 일반적인 학습 과정은 다음과 같습니다:

학습 가능한 매개변수(또는 가중치(weight))를 갖는 신경망을 정의합니다.

데이터셋(dataset) 입력을 반복합니다.

입력을 신경망에서 전파(process)합니다.

손실(loss; 출력이 정답으로부터 얼마나 떨어져있는지)을 계산합니다.

변화도(gradient)를 신경망의 매개변수들에 역으로 전파합니다.

신경망의 가중치를 갱신합니다. 일반적으로 다음과 같은 간단한 규칙을 사용합니다: 새로운 가중치(weight) = 가중치(weight) - 학습률(learning rate) * 변화도(gradient)

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 입력 이미지 채널 1개, 출력 채널 6개, 3x3의 정사각 컨볼루션 행렬
        # 컨볼루션 커널 정의
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        # affine calculation : y = Wx + b
        self.fc1 = nn.Linear(16*6*6, 120) # 6*6 is image dimension
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)


    def forward(self,x):
        # (2,2)
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        # 크기가 제곱수라면 하나의 숫자만을 특정
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def num_flat_features(self,x):
        size = x.size()[1:]  # 배치 차원을 제외한 모든 차원
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)



