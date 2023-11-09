import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

hidden_units = 5120
dropout_rate = 0.5

class net(nn.Module):
    def __init__(self , model):
        super(net, self).__init__()
        self.net_layer = torch.nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.net_layer(x)
        return x.view(x.shape[0: 2])

class net2(nn.Module):
    def __init__(self, class_num):
        super(net2, self).__init__()
        self.fc = nn.Linear(hidden_units, class_num)

    def forward(self, x):
        x = self.fc(x)
        x = F.dropout(x, dropout_rate, training = self.training)
        return x

def extract_feature(data, model_name):
    # if model_name == "resnet50":
    #     p = 2048
    #     MODEL = models.resnet50(pretrained = True)
    # elif model_name == "googlenet":
    #     p = 1024
    #     MODEL = models.googlenet(pretrained = True)
    # elif model_name == "resnext":
    #     p = 2048
    #     MODEL = models.resnext50_32x4d(pretrained = True)
    
    # for param in MODEL.parameters():
    #     param.requires_grad = False
    
    model = net(model_name)
    model.eval()

    with torch.no_grad():
        feature = model(data)

    # print(feature.shape)
    # print(label.shape)
    return feature

def model_fusion(data, label, num_labels):
     
    #  feature = extract_feature(data, label, "resnet50")
    #  feature += extract_feature(data, label, "googlenet")
    #  feature += extract_feature(data, label, "resnext")

    for model_name in ["resnet50", "googlenet", "resnext"]:
        feature = extract_feature(data, model_name)
        features = torch.cat((features, feature), 1) if model_name != "resnet50" else feature
    

