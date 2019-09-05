from modules import segmentation_models as smp
import torch
import torch.nn as nn
from pytorch_modelsize import SizeEstimator 
class Model(nn.Module):
    def __init__(self, module_name, num_class=4):
        super(Model, self).__init__()
        self.module_name = module_name
        self.modules = {
            "Unet_Resnet34": smp.Unet('resnet34', classes=num_class, activation='softmax'),
            "Linknet_Resnet34": smp.Linknet('resnet34', classes=num_class, activation='softmax'),
            "FPN_Resnet34": smp.Unet('resnet34', classes=num_class, activation='softmax'),
            "PSPNet_Resnet34": smp.PSPnet('resnet34', classes=num_class, activation='softmax'),
        }
    
    def forward(self, inputs):
        outputs = self.modules[self.module_name](inputs)
        return outputs 
# from torchvision import models

# inputs = torch.randn(5, 3, 256, 1600)
# model = smp.Unet('resnet34', classes=4, activation='softmax')
# se = SizeEstimator(model, input_size=(2, 3, 256, 256))
# print(se.estimate_size())