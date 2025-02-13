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

def add_noise_timesteps(raw_data,decline_rate,noise_rate,seed,timesteps):
    # 因为Diffusion模型是多次迭代的，所以这里我们需要多次处理
    pic = raw_data
    for i in range(timesteps):
        pic = add_noise(pic,decline_rate,noise_rate,seed)
    return pic
# 上面是基本原理，然后底下我们上numpy加速

def AddNoise(raw_data, decline_rate, noise_rate, seed):
    np.random.seed(seed)
    noise = np.random.normal(0, 1, raw_data.shape)
    pic = raw_data * decline_rate + noise_rate * noise
    return pic

def AddNoise_timesteps(raw_data, decline_rate, noise_rate, seed, timesteps):
    pic = raw_data
    for _ in range(timesteps):
        pic = add_noise(pic, decline_rate, noise_rate, seed)
    return pic

# 然后我们再换成torch版本，GPU is all we need.