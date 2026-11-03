import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
x = torch.randn(1000, 1000, device="cuda")
print((x @ x).sum())