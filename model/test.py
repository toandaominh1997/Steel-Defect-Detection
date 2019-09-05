import torchvision
import torch
model = torchvision.models.segmentation.fcn_resnet50()

inputs = torch.randn(1, 3, 256, 256)
# model = Unet("resnet18", encoder_weights="imagenet", classes=4, activation=None)

outputs = model(inputs)
print(outputs)
    

print(outputs['out'].size())