from timm.models.ghostnet import ghostnetv2_130, ghostnet_130, GhostModuleV2
from timm.models.resnet import resnet50
from ghostnet3d import GhostModule3D, GhostBottleneck3D, GhostModule3DV2
import torch
import torch.nn as nn
from calflops import calculate_flops

# model = ghostnetv2_130(num_classes=99)
# model = ghostnet_130(num_classes=99)
# model = resnet50(num_classes=99)
# model = GhostModuleV2(3, 3, act_layer=nn.Hardsigmoid)
# model = GhostModule3D(17, 32, act_layer=nn.Hardsigmoid)
# model = GhostModule3DV2(17, 32, act_layer=nn.Hardsigmoid)
# model = GhostBottleneck3D(17, 17//2, 32, act_layer=nn.Hardsigmoid)
# model = nn.Sequential(
#     GhostBottleneck3D(17, 17//2, 32, act_layer=nn.Hardsigmoid),
#     GhostBottleneck3D(32, 32//2, 64, act_layer=nn.Hardsigmoid),
#     GhostBottleneck3D(64, 64//2, 128, act_layer=nn.Hardsigmoid),
#     GhostBottleneck3D(128, 128//2, 256, act_layer=nn.Hardsigmoid),
#     nn.AdaptiveAvgPool3d((1, 1, 1)),
# )
model = nn.Sequential(
    GhostModule3DV2(17, 32, act_layer=nn.Hardsigmoid),
    nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
    GhostModule3DV2(32, 128, act_layer=nn.Hardsigmoid),
    nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
    GhostModule3DV2(128, 256, act_layer=nn.Hardsigmoid),
    nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
    GhostModule3DV2(256, 512, act_layer=nn.Hardsigmoid),
    nn.AdaptiveAvgPool3d((1, 1, 1)),
)
batch_size = 1
input_size = (batch_size, 17, 32, 56, 56)
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_size,
                                      output_as_string=True,
                                      print_results=False,
                                      output_precision=4)
print("GhostNet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

for k, v in model.state_dict().items():
    print(k, v.shape)

output = model(torch.randn(1, 17, 32, 56, 56))
print(output.shape)