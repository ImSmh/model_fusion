'''
Author: smh smh0240@163.com
Date: 2023-11-08 20:14:33
LastEditors: smh smh0240@163.com
LastEditTime: 2023-11-08 20:27:41
FilePath: \model_fusion\getData.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

def getData(dir, test_dir, batch_size, transform, shuffle):
    train_data = ImageFolder(dir)

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 2)

