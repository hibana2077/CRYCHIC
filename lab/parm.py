from timm.models.ghostnet import ghostnetv2_130, ghostnet_130, GhostModuleV2
from timm.models.resnet import resnet50
import torch
import torch.nn as nn
from calflops import calculate_flops

# model = ghostnetv2_130(num_classes=99)
# model = ghostnet_130(num_classes=99)
# model = resnet50(num_classes=99)
model = GhostModuleV2(3, 3, act_layer=nn.Hardsigmoid)
batch_size = 1
input_size = (batch_size, 3, 224, 224)
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_size,
                                      output_as_string=True,
                                      print_results=False,
                                      output_precision=4)
print("GhostNet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

for k, v in model.state_dict().items():
    print(k, v.shape)