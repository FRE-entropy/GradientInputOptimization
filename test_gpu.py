import torch

print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU设备数:", torch.cuda.device_count())
    print("当前GPU设备:", torch.cuda.current_device())
    print("GPU名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("未检测到GPU，将使用CPU")