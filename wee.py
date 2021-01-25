import torch
from torch.autograd import Variable

x = Variable(torch.ones(10), requires_grad=True)
y = x * Variable(torch.linspace(1, 10, 10), requires_grad=False)
y.backward(torch.ones(10))
print(y)