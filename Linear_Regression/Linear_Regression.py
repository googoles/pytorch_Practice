import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
# 1 Train할 변수 선언

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

print(x_train)
print(x_train.shape)

print(y_train)
print(y_train.shape)
# 가중치와 편향의 초기화

W = torch.zeros(1, requires_grad=True)
# requires_grad = 학습을 통해 값이 변하는 변수
print(W)

b = torch.zeros(1,requires_grad=True)
print(b)

# 가설 세우기
# H(x) = x_train * W + b

hypothesis = x_train * W + b
print(hypothesis)
# cost function

cost = torch.mean((hypothesis - y_train) ** 2)
print(cost)

# 경사하강법 구현(SGD)

optimizer = optim.SGD([W, b], lr = 0.01)

optimizer.zero_grad() # gradient를 0으로 초기화 = 기울기 0
cost.backward() # 비용함수를 미분하여 gradient 계산
optimizer.step() # W, b 업데이트


nb_epochs = 4000 # 원하는 만큼 경사 하강법 반복

for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train * W + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))