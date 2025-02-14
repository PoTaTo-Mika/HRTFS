import numpy as np
import logging
import random

# 这里是对2D数组进行处理的函数 [行，列]
# array = np.arrary([[1,2,3],
#                    [4,5,6],
#                    [7,8,9]])

def add_noise(raw_data,decline_rate,noise_rate,seed):
    # 伟大的DDPM指引着我们
    # Xi =  decline_rate * X_{i-1} + noise_rate * N(0,1)
    pic = raw_data 
    # 这里先读取原始数据
    # logging.info(pic.shape())
    for i in range(pic.shape[0]):
        # 因为我们并非处理绘画图片，而是处理大气数据（已经是数组了）
        # 所以不用考虑RGB通道
        # logging.info(pic[i].shape())
        for j in range(pic.shape[1]):
            # 假设是行*列的二维数组
            pic[i][j] = pic[i][j] * decline_rate + noise_rate * random.gauss(0,1)
    # 返回处理后的数据
    return pic

# 上面是基本原理，然后底下我们上numpy加速

def AddNoise(raw_data, decline_rate, noise_rate, seed):
    np.random.seed(seed)
    noise = np.random.normal(0, 1, raw_data.shape)
    pic = raw_data * decline_rate + noise_rate * noise
    return pic

# 虽然忙了半天，但是忙了半天，接下来torch才是我们会使用的版本。

import torch

def gen_rate(generator, steps):
    # 我们要确保这里生成的是一个2*steps的Tensor
    rate_list = generator(steps)
    return rate_list

def ddpm(raw_data, generator, seed, steps):
    # 生成decline_rate列表和noise_rate列表
    # rate_list = [decline_rate -> ndarray, noise_rate -> ndarray]
    torch.manual_seed(seed)
    rate_list = gen_rate(generator, steps)
    # X{i} =  decline_rate * X_{i-1} + noise_rate * N(0,1)
    X = raw_data.clone()
    for i in range(steps):
        X = X * rate_list[0][i] + rate_list[1][i] * torch.normal(0, 1, X.shape)
    return X # Tensor

if __name__ == '__main__':
    from noise_scheduler import linear_scheduler
    a = torch.Tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    b = ddpm(a, linear_scheduler, 0, 4)
    print(b)
