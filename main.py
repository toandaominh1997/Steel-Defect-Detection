from modelss import ModelBuilder, SegmentationModule

net_encoder = ModelBuilder.build_encoder(
        arch="hrnetv2",
        fc_dim=720,
        weights='')
net_decoder = ModelBuilder.build_decoder(
    arch="c1",
    fc_dim=720,
    num_class=4,
    weights='')

import torch 
inputs = torch.randn(2, 3, 64, 128)
print(inputs.size())
outputs = net_encoder(inputs, return_feature_maps=True)
# outputs = net_decoder(net_encoder(inputs, return_feature_maps=True))
outputs = net_decoder(outputs)
print(outputs.size())
# print(pred_deepsup.size())

