import torch

x = torch.ones(2,2,requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

a = torch.randn(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad) # 기존 Tensor의 requires_grad 값을 바꿔치기 (in-place)하여 변경합니다. 입력값이 지정되지 않으면 기본값은 False 입니다.

a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)

# Gradient 변화도
# 역전파 (backdrop)

out.backward()
print(x.grad)

# torch.autograd = Vector-jacobian Matrix을 계산하는 엔진이다.

# Vector_Jacobian Matrix
x = torch.randn(3, requires_grad=True)
y = x*2
while y.data.norm() < 1000:
    y = y * 2

print(y)

# torch.autograd = torch.autograd 는 전체 야코비안을 \
# 직접 계산할수는 없지만, 벡터-야코비안 곱은 간단히 \
# backward 에 해당 벡터를 인자로 제공하여 얻을 수 있습니다:
v = torch.tensor([0.1,1.0,0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad(): # autograd가 .requires_grad=True인 \
    # Tensor들의 연산기록을 추적하는것을 멈춘다.
    print((x**2).requires_grad)

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())