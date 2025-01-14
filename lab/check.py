import torch
# model_pth = '../posec3d/posec3d/slowonly_r50_gym/joint/best_top1_acc_epoch_5.pth'
model_pth = './X3D-shallow.pth'

# 加載模型的 state_dict
checkpoint = torch.load(model_pth, map_location='cpu')
# print(checkpoint.keys())
state_dict = checkpoint

# 計算參數總數
total_params = sum(p.numel() for p in state_dict.values())
print(f"Total number of parameters(M): {total_params/1e6:.2f}M")

# print full model
for k, v in state_dict.items():
    print(k, v.shape)

# print(checkpoint['meta'])

# model = state_dict

# batch_size = 32
# input_size = (batch_size, 17, 1, 7, 7)
# flops, macs, params = calculate_flops(model=model, 
#                                       input_shape=input_size,
#                                       output_as_string=True,
#                                       output_precision=4)
# print("GhostNet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))