import torch
from torch.utils.data import DataLoader, Dataset

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import LEVIR_CD_Dataset, SVCD_Dataset
from change_detection_pytorch.utils.lr_scheduler import GradualWarmupScheduler
from change_detection_pytorch.utils.utils import seed_everything

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定随机种子，保证每次训练流程尽可能可复现
SEED = 42
seed_everything(SEED, workers=True, deterministic=True)

# 原始配置（保留作为对照，不删除）
# model = cdp.Unet(
#     encoder_name="resnet34",  # 原始主干：ResNet34
#     encoder_weights="imagenet",  # ImageNet 预训练权重
#     in_channels=3,
#     classes=2,
#     siam_encoder=True,
#     fusion_form='concat',
# )

# 新配置：仅替换 Encoder 为 MobileNetV2（收益高、改动小）
# 说明：
# 1) 仍然使用 UNet 解码器与原训练流程，确保改动范围只在“左半边特征提取主干”。
# 2) encoder_name="mobilenet_v2" 对应项目内注册的 MobileNetV2Encoder，
#    其预训练权重链接指向 PyTorch 官方发布的 mobilenet_v2 权重。
# 3) 这样做可以利用轻量主干在参数量与推理速度上的优势，同时保持你现有代码结构稳定。
model = cdp.Unet(
    encoder_name="mobilenet_v2",   # 替换为 MobileNetV2 主干
    encoder_weights="imagenet",    # 使用 PyTorch 官方 ImageNet 预训练权重
    in_channels=3,                   # 输入通道数（RGB=3）
    classes=2,                       # 输出类别数（二分类变化检测）
    siam_encoder=True,               # 保持 Siamese 编码结构不变
    fusion_form='concat',            # 保持双时相特征融合方式不变
)

# model = cdp.STANet(
#     encoder_name="resnet18",     # 必须明确指定带层数的具体型号
#     encoder_weights="imagenet",  # 挂上预训练权重，加速收敛
#     in_channels=3,  # 输入通道数（如 RGB 图像为 3）
#     classes=2,      # 输出通道数（如二分类为 2）
#     siam_encoder=True,  # 是否使用 Siamese 编码器
#     fusion_form='concat',  # 特征融合方式（如 concat, sum, diff, abs_diff）
# )

print("正在使用 UNet + MobileNetV2(预训练) 模型进行训练...")

train_dataset = LEVIR_CD_Dataset(r'D:\DeepLearning\dataset\LEVIR\train_cropped',
                                 sub_dir_1='A',
                                 sub_dir_2='B',
                                 img_suffix='.png',
                                 ann_dir=r'D:\DeepLearning\dataset\LEVIR\train_cropped\label',
                                 debug=False)

valid_dataset = LEVIR_CD_Dataset(r'D:\DeepLearning\dataset\LEVIR\test_cropped',
                                 sub_dir_1='A',
                                 sub_dir_2='B',
                                 img_suffix='.png',
                                 ann_dir=r'D:\DeepLearning\dataset\LEVIR\test_cropped\label',
                                 debug=False,
                                 test_mode=True)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

#loss = cdp.utils.losses.CrossEntropyLoss()
loss1 = cdp.utils.losses.CrossEntropyLoss()
loss2 = cdp.losses.DiceLoss(mode='multiclass')
loss = cdp.losses.HybridLoss(loss1, loss2, reduction='sum')

metrics = [
    cdp.utils.metrics.Fscore(activation='argmax2d'),
    cdp.utils.metrics.Precision(activation='argmax2d'),
    cdp.utils.metrics.Recall(activation='argmax2d'),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, ], gamma=0.1)

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = cdp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = cdp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# train model for 60 epochs

max_score = 0
MAX_EPOCH = 60

for i in range(MAX_EPOCH):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    scheduler_steplr.step()

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['fscore']:
        max_score = valid_logs['fscore']
        print('max_score', max_score)
        torch.save(model, './best_model.pth')
        print('Model saved!')

# save results (change maps)
"""
Note: if you use sliding window inference, set: 
    from change_detection_pytorch.datasets.transforms.albu import (
        ChunkImage, ToTensorTest)
    
    test_transform = A.Compose([
        A.Normalize(),
        ChunkImage({window_size}}),
        ToTensorTest(),
    ], additional_targets={'image_2': 'image'})

"""
valid_epoch.infer_vis(valid_loader, save=True, slide=False, save_dir='./res')
# print(f"Number of training samples: {len(train_dataset)}")
# for i in range(len(train_dataset)):
#     sample = train_dataset[i]
#     print(f"Sample {i}: {sample}")
#     break  # 只打印第一个样本