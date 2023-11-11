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
    


class fusionNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        # self.resnet = models.resnet50(pretrained = True)
        # self.google = models.googlenet(pretrained = True)
        # self.resnext = models.resnext50_32x4d(pretrained = True)
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.google = models.googlenet(weights='IMAGENET1K_V1')
        self.resnext = models.resnext50_32x4d(weights='IMAGENET1K_V1')
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.google.parameters():
            param.requires_grad = False
        for param in self.resnext.parameters():
            param.requires_grad = False
        num_features = self.resnet.fc.in_features + self.google.fc.in_features + self.resnext.fc.in_features
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[: -2])
        self.google = torch.nn.Sequential(*list(self.google.children())[: -3])
        self.resnext = torch.nn.Sequential(*list(self.resnext.children())[: -2])

        # in_channels = 1024 + 2048 + 2048
        # out_channels = 4096
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_channels)   

        # self.conv3 = nn.Conv2d(9216, out_channels // 2, kernel_size=3, stride = 2, padding = 1)   
        # self.conv4 = nn.Conv2d(out_channels // 2, out_channels // 8, kernel_size = 3, stride = 2, padding = 1)   
        # self.fc = nn.Linear(out_channels // 2, num_classes)

        in_channels = 5120
        out_channels = 1024
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(6144, 2048, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(2048)
        self.conv4 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 2 * 2, num_classes)
    
    def forward(self, x):
        resnet_features = self.resnet(x)
        google_features = self.google(x)
        resnext_features = self.resnext(x)
        combined_features = torch.cat((resnet_features, google_features, resnext_features), dim = 1)
        # combined_features = combined_features.view(combined_features.size(0), -1)

        # residual = combined_features
        # out = self.conv1(combined_features)
        # out = self.bn1(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # out = torch.cat((out, residual), dim = 1)

        # out = self.conv3(out)
        # out = self.relu(out)
        # out = self.conv4(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)

        identity = combined_features
        out = self.conv1(combined_features)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.cat((out, identity), dim = 1)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.fc(out.view(out.size(0), -1))

        return out
