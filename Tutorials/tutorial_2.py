# Autograd : Automatic differential Equation

import torch

x = torch.ones(2,2, requires_grad=True)
print(x)
# Tensor 연산
y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z,out)

# inplace Tensor's requires_grad value if it is not defined
# Default value is False

a = torch.randn(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)

# Gradient
# Now I started to backprop
# torch.autograd = Vector-Jacobian Matrix

out.backward()
print(x.grad)

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() <1000:
    y = y * 2

print(y)

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x+y
    print(z)
    print(z.to('cpu',torch.double))

