import os
import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from tqdm import tqdm

def batch_visualize_slic():
    # 1. 设置好你的输入输出路径
    dir_A = r"D:\DeepLearning\dataset\LEVIR\val_cropped\A"
    dir_B = r"D:\DeepLearning\dataset\LEVIR\val_cropped\B"
    
    # 你要的专属文件夹
    dir_slic_out = r"D:\DeepLearning\change_detection.pytorch\SLIC\for_people"
    os.makedirs(dir_slic_out, exist_ok=True) # 如果没有这个文件夹，自动创建

    # 获取这 1024 张图的文件名
    img_names = [f for f in os.listdir(dir_B) if f.endswith('.png')]
    print(f"总共找到 {len(img_names)} 张图片，开始批量切图并画线...")

    # 2. 开始循环处理
    for name in tqdm(img_names):
        path_a = os.path.join(dir_A, name)
        path_b = os.path.join(dir_B, name)
        path_out = os.path.join(dir_slic_out, name)

        # 读图并转 RGB
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        if img_a is None or img_b is None:
            continue
            
        img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        # 拼接并分割 (保持 compactness=1 极致贴合边缘)
        img_concat = np.concatenate((img_a_rgb, img_b_rgb), axis=-1)
        segments = slic(img_concat, n_segments=300, compactness=1, channel_axis=-1)

        # 把黄线画在 B 图上
        vis_img = mark_boundaries(img_b_rgb, segments, color=(1, 1, 0))
        
        # 还原成 uint8 和 BGR 格式用来保存
        vis_img_save = (vis_img * 255).astype(np.uint8)
        vis_img_save = cv2.cvtColor(vis_img_save, cv2.COLOR_RGB2BGR)
        
        # 存入 SLIC 文件夹
        cv2.imwrite(path_out, vis_img_save)

    print(f"\n搞定！1024 张网格图已经全部存入: {dir_slic_out}")

if __name__ == '__main__':
    batch_visualize_slic()