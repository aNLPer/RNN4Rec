import torch
from torch.distributions.categorical import Categorical

l = torch.nn.NLLLoss()
pred = torch.Tensor([[-1.0, -1.0],
        [-7.0, -7.0],
        [-3.0, -3.0]])
target = torch.Tensor([0, 1, 0]).long()

print(l(pred, target))