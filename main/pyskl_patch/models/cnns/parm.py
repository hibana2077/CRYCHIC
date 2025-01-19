from ghostnet3d import GhostModule3D, GhostBottleneck3D, GhostModule3DV2, GhostNet3D
import torch
import torch.nn as nn
from calflops import calculate_flops

# SiLU GhostNet3D FLOPs:1.0068 GFLOPS   MACs:480.575 MMACs   Params:283.808 K
# HardSigmoid GhostNet3D FLOPs:997.954 MFLOPS   MACs:480.575 MMACs   Params:283.808 K
# ReLU GhostNet3D FLOPs:1.0068 GFLOPS   MACs:480.575 MMACs   Params:283.808 K
# ReLU6 GhostNet3D FLOPs:997.954 MFLOPS   MACs:480.575 MMACs   Params:283.808 K
# LeakyReLU GhostNet3D FLOPs:1.0068 GFLOPS   MACs:480.575 MMACs   Params:283.808 K
# Hardtanh GhostNet3D FLOPs:997.954 MFLOPS   MACs:480.575 MMACs   Params:283.808 K
# Hardsiwsh GhostNet3D FLOPs:997.954 MFLOPS   MACs:480.575 MMACs   Params:283.808 K
model = nn.Sequential(
    GhostModule3D(in_chs=17, out_chs=32, kernel_size=3, stride=1, act_layer=nn.ReLU),
    nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
    GhostModule3D(in_chs=32, out_chs=64, kernel_size=3, stride=1, act_layer=nn.ReLU),
    nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
    GhostBottleneck3D(in_chs=64, mid_chs=128, out_chs=256),
    nn.AdaptiveAvgPool3d((1, 1, 1)),
)

# (3,1,1,1) FLOPs:2.2438 GFLOPS   MACs:1.106 GMACs   Params:102.624 K
# (3,2,2,2) FLOPs:4.0365 GFLOPS   MACs:2.0009 GMACs   Params:554.208 K
# (3,3,3,3) FLOPs:7.1648 GFLOPS   MACs:3.5665 GMACs   Params:1.7799 M
# (3,5,3,3) FLOPs:13.4589 GFLOPS   MACs:6.7135 GMACs   Params:2.081 M
# (5,1,1,1) FLOPs:8.9313 GFLOPS   MACs:4.4498 GMACs   Params:182.592 K
# (7,1,1,1) FLOPs:23.8074 GFLOPS   MACs:11.8878 GMACs   Params:360.48 K
# model = GhostNet3D(in_channels=17, base_channels=32, num_stages=4, up_strides=(1, 2, 2, 2), pool_strat="avg", act_layer=['ReLU'])
batch_size = 1
input_size = (batch_size, 17, 32, 56, 56)
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_size,
                                      output_as_string=True,
                                      print_results=False,
                                      output_precision=4)
print("GhostNet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

# for k, v in model.state_dict().items():
#     print(k, v.shape)

output = model(torch.randn(1, 17, 32, 56, 56))
print(output.shape)