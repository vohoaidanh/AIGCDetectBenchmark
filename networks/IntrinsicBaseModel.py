# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from networks.resnet import resnet50
#import torch.utils.model_zoo as model_zoo

def resnet50_fusion(num_classes=1,pretrained=True):
    model = Intrinsic_Clf(num_classes, pretrained=True)
    return model

class Intrinsic_Clf(nn.Module):
    def __init__(self, num_classes,pretrained=True):
        super(Intrinsic_Clf,self).__init__()
        self.origin = resnet50(pretrained=True)
        self.shading = resnet50(pretrained=True)
        
        #model.fc = nn.Linear(2048, 1)
        #torch.nn.init.normal_(model.fc.weight.data, 0.0, opt.init_gain)
        num_ftrs1 = self.origin.fc.in_features
        self.origin.fc = nn.Linear(num_ftrs1, 1024)  # Đầu ra của nhánh 1
        
        num_ftrs2 = self.shading.fc.in_features
        self.shading.fc = nn.Linear(num_ftrs2, 1024)  # Đầu ra của nhánh 2
        
        # Head classification
        self.classification_head = nn.Sequential(
            nn.Linear(2048, num_classes),  # 512 là tổng số features từ hai nhánh
            #nn.ReLU(inplace=True),
            #nn.Linear(256, num_classes)  # Số lượng lớp đầu ra
        )
    
    def forward(self, x1, x2):
        #xi is GRB image origin
        #x2 is shading image
        x1 = self.origin(x1)
        x2 = self.shading(x2)
        
        # Kết hợp hai features từ hai nhánh
        x = torch.cat((x1, x2), dim=1)
        
        # Head classification
        x = self.classification_head(x)
        return x
    
    
    
    
    
    
    
    
    
    
    