import torch

print("----------is cuda available?", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
print("current device ", torch.cuda.current_device())