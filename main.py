'''
Author: smh smh0240@163.com
Date: 2023-11-08 09:39:12
LastEditors: smh smh0240@163.com
LastEditTime: 2023-11-11 14:52:18
FilePath: \model_fusion\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
from getData import getData
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from net import model_fusion
from net import extract_feature
from net import net2
import torch.nn as nn
import torchvision.models as models
from net import fusionNet

# train_dir = "E:\\datasets\\cat_dog\\training_set\\training_set"
train_dir = "E:\\datasets\\flowers\\train"
# test_dir = "E:\\datasets\\cat_dog\\test_set\\test_set"
test_dir = "E:\\datasets\\flowers\\tmp"
MEAN = [0.485, 0.456, 0.406] # ImageNet数据集的均值
STD = [0.229, 0.224, 0.225] # ImageNet数据集的标准差
batch_size = 8
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = MEAN, std = STD)
])
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 30

# 1. 加载数据集
train_data = ImageFolder(train_dir, transform = transform)
test_data = ImageFolder(test_dir, transform = transform)

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 2)
test_loader = DataLoader(test_data, batch_size = 1, shuffle = True, pin_memory = True, num_workers = 2)

model = fusionNet(len(train_data.classes))
for param in model.parameters():
    param.requires_grad = True
# model = models.resnet50(weights='IMAGENET1K_V1')
model.to(device)
# resnet = models.resnet50(pretrained = True).to(device)
# googlenet = models.googlenet(pretrained = True).to(device)
# resnext = models.resnext50_32x4d(pretrained = True).to(device)
resnet = models.resnet50(weights='IMAGENET1K_V1').to(device)
googlenet = models.googlenet(weights='IMAGENET1K_V1').to(device)
resnext = models.resnext50_32x4d(weights='IMAGENET1K_V1').to(device)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)
loss_func = nn.CrossEntropyLoss()

def train(epoch):
    # for epoch in range(epochs):
    model.train()
    total_loss = 0.
    for idx, (data, label) in enumerate(train_loader):
        # model_fusion(data, label, len(train_data.classes))
        # for model_name in ["resnet50", "googlenet", "resnext"]:
        
        data = data.to(device)
        label = label.to(device)

        # for model_name in [resnet, googlenet, resnext]:
        #     feature = extract_feature(data, model_name)
        #     features = torch.cat((features, feature), 1) if model_name != resnet else feature
        
        # features = features.to(device)
        # label = label.to(device)
        
        output = model(data)
        loss = loss_func(output, label)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    print("Epoch: [{} / {}], Loss: {:.6f}".format(epoch + 1, epochs, total_loss / len(train_loader)))
        
def test():
    test_loss = 0.
    correct = 0.
    model.eval()
    with torch.no_grad():
        for idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)

            # for model_name in [resnet, googlenet, resnext]:
            #     feature = extract_feature(data, model_name)
            #     features = torch.cat((features, feature), 1) if model_name != resnet else feature
            
            output = model(data)
            test_loss += loss_func(output, label)
            pred = output.argmax(dim = 1)
            correct += pred.eq(label.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test Loss: {:.4f}, Accuracy: [({} / {}): {:.2f}%]".format(test_loss, correct, len(test_loader.dataset) , float(correct) / len(test_loader.dataset) * 100))


if __name__ == "__main__":
    for epoch in range(epochs):
        train(epoch)
        test()