import torch

tensor = torch.tensor([[1, 3, 2], [8, 5, 6]])
print(torch.max(tensor, dim=1)[1])