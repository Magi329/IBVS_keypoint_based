## 把图片里的物体分割出来，其余部分纯黑 ##

from florence_sam import *
import PIL.Image as Image
import os
import re
import h5py
import numpy as np
import pandas as pd

START_INDEX = 60
END_INDEX = 125

def extract_number(filename):
    # 使用正则表达式提取文件名中的数字
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')  # 没有数字返回无穷大

def add_mask(i, input_image, mask):
    # 把input_image的mask以外的部分设置成黑色 
    input_image_np = np.array(input_image)
    mask_np = np.array(mask[0]).astype(bool)

    black_image = np.zeros_like(input_image_np)
    black_image[mask_np] = input_image_np[mask_np]
    result_image = Image.fromarray(black_image)

    result_image_path = f'/home/chen/Desktop/episode2/{i}.png'
    result_image.save(result_image_path)
    print(f"Result image saved at: {result_image_path}")

    return result_image


# 定义处理文件的路径
base_dir = '/home/chen/Desktop/Pengfei/data/episode_2/images/cam_left_hand_rs'
photos = sorted(
    [os.path.join(base_dir, photo) for photo in os.listdir(base_dir)],
        key=lambda x: int(extract_number(os.path.basename(x)))  # 按顺序读取photo
)
# 筛选序号在 60 到 125 之间的图片
selected_photos = [
    photo for photo in photos
    if START_INDEX <= extract_number(os.path.basename(photo)) <= END_INDEX
]
# print(selected_photos)
# 定义prompt
PROMPT = 'fruit'


# 批量处理loop：
for i, photo in enumerate(selected_photos):
    # 读取图片
    input_image = Image.open(photo)
    # 创建实例，输入 prompt 给florence——sam函数
    if 'Seg' not in locals():
        Seg = Segmentation()
    annotated_image, mask = Seg.process_image(input_image, PROMPT)
    # 加mask, 存照片
    masked_image = add_mask(i, input_image, mask)









# with h5py.File('episodes_masked_data.h5', 'r') as f:
#     # 读取特定 episode 的数据
#     episode_images = f['episode_2/images'][:]
#     episode_observations = f['episode_2/observations'][:]
    
#     # 提取第 60 到 125 步的图片，注意索引是从 0 开始的
#     start_step = 60
#     end_step = 125
#     selected_images = episode_images[start_step:end_step+1]  # 包含第125步
#     print(f"Selected images shape: {selected_images.shape}")

#     # 提取第 60 到 125 步的qpos
#     qpos_array = episode_observations[start_step:end_step+1]
#     print(f"qposs shape: {qpos_array.shape}")

#     # add mask
#     for idx, image in enumerate(selected_images):
#         selected_images[idx] = mask_image(image, 'fruit')


# input_image_path = '/home/chen/Desktop/bn_color_image.png'
# input_image = Image.open(input_image_path)
# prompt_text = 'mouse'

# Seg = Segmentation()
# annotated_image, mask = Seg.process_image(input_image, prompt_text)

# 保存图片
# annotated_image_path = '/home/chen/Desktop/annotated_image.png'
# annotated_image.save(annotated_image_path)
# mask_image_path = '/home/chen/Desktop/mask_image.png'
# mask_image = Image.fromarray(mask)
# mask_image.save(mask_image_path)






