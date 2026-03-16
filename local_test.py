import torch
from torch.utils.data import DataLoader, Dataset

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import LEVIR_CD_Dataset, SVCD_Dataset
from change_detection_pytorch.utils.lr_scheduler import GradualWarmupScheduler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = cdp.Unet(
#     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=2,  # model output channels (number of classes in your datasets)
#     siam_encoder=True,  # whether to use a siamese encoder
#     fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
# )

model = cdp.STANet(
    encoder_name="resnet34",     # 必须明确指定带层数的具体型号
    encoder_weights="imagenet",  # 挂上预训练权重，加速收敛
    in_channels=3,  # 输入通道数（如 RGB 图像为 3）
    classes=2,      # 输出通道数（如二分类为 2）
    siam_encoder=True,  # 是否使用 Siamese 编码器
    fusion_form='concat',  # 特征融合方式（如 concat, sum, diff, abs_diff）
)

print("正在使用 STANet 模型进行训练...")

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

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

loss = cdp.utils.losses.CrossEntropyLoss()
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