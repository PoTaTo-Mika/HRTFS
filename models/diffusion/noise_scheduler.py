import torch

def linear_scheduler(timesteps):
    # 返回一个线性下降的decline_rate和线性上升的noise_rate
    decline_rate = torch.linspace(1, 0, timesteps)
    noise_rate = torch.linspace(0, 1, timesteps)
    return [decline_rate, noise_rate]

def cosine_scheduler(timesteps):
    # 返回一个余弦下降的decline_rate和余弦上升的noise_rate
    decline_rate = torch.cos(torch.linspace(0, torch.pi / 2, timesteps))
    noise_rate = torch.sin(torch.linspace(0, torch.pi / 2, timesteps))
    return [decline_rate, noise_rate]