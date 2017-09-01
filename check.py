import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
b = torch.Tensor(np.ones((1,1,3,3)))
b = Variable(b, requires_grad=True)
con = nn.Conv2d(1,1,3)
c = con(b)
opt = torch.optim.SGD(con.parameters(),lr=0.1)
opt.zero_grad()
c.backward(retain_graph=True)

#opt.step()
print dir(b)
print(b.grad)
b.grad.data.zero_()
opt.zero_grad()
c.backward()
opt.step()
print(b.grad)
