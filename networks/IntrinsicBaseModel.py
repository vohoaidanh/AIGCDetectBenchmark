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
        
        self.origin.fc = nn.Identity()
        self.shading.fc = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(2048 * 2, 512), 
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x1, x2):
        #xi is GRB image origin
        #x2 is shading image
        
        # Forward pass through the first ResNet-50 model
        features1 = self.origin(x1)
        # Forward pass through the second ResNet-50 model
        features2 = self.shading(x2)
        
        # Kết hợp hai features từ hai nhánh
        features  = torch.cat((features1, features2), dim=1)
        output = self.head(features)
        return output
    
    
    
    
    
    
    
    
    
    
    