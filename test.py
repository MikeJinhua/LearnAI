import torch
device = "mps"

x = torch.randn(1000,1000).to(device)
y = torch.matmul(x,x)

print(y)