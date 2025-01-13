from timm.models.ghostnet import ghostnetv2_130, ghostnet_130
from timm.models.resnet import resnet50
import torch
from calflops import calculate_flops

# model = ghostnetv2_130()
# model = ghostnet_130()
model = resnet50()
batch_size = 1
input_size = (batch_size, 3, 224, 224)
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_size,
                                      output_as_string=True,
                                      output_precision=4)
print("GhostNet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))