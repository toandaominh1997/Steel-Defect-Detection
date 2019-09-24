import torch
import torch.nn as nn
from models.model_cls import Resnet34Classification
from modules import segmentation_models as smp
from modules import ModelBuilder

class Model(nn.Module):
    def __init__(self, 
            num_class=4, 
            mode='cls', 
            encoder='resnet34', 
            decoder='Unet', 
            activation='softmax', 
            encoder_weights='imagenet'):
        super(Model, self).__init__()
        net_encoder = ModelBuilder.build_encoder(
                arch="hrnetv2",
                fc_dim=720,
                weights='')
        net_decoder = ModelBuilder.build_decoder(
            arch="c1",
            fc_dim=720,
            num_class=num_class,
            weights='')
        if mode=='cls':
            self.models = Resnet34Classification(num_class=num_class)
        else:
            if decoder=='Unet':
                print('Unet')
                self.models = smp.Unet(encoder_name=encoder, classes=num_class, activation=activation, encoder_weights=encoder_weights)
            elif decoder=='Linknet':
                self.models = smp.Linknet(encoder_name=encoder, classes=num_class, activation=activation, encoder_weights=encoder_weights)
            elif decoder=='FPN':
                self.models = smp.FPN(encoder_name=encoder, classes=num_class, activation=activation, encoder_weights=encoder_weights)
            elif decoder=='PSPNet':
                self.models = smp.PSPNet(encoder_name=encoder, classes=num_class, activation=activation, encoder_weights=encoder_weights)
            elif decoder=='hrnet':
                print('HRNET')
                self.models = nn.Sequential(net_encoder, net_decoder)
            
    def forward(self, inputs):
        return self.models(inputs)


