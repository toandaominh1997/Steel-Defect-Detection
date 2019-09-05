import segmentation_models as smp 
import torch 
model = smp.Unet('vgg11', classes = 4, encoder_weights='imagenet')

model = smp.PSPNet('vgg11', classes=4, encoder_weights='imagenet')
inputs = torch.randn(5, 3, 256, 1600)
outputs = model(inputs)
print(model.__class__.__name__)

print(outputs.size())