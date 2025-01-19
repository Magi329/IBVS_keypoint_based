import os  
import cv2 
import torch
import numpy as np  
import threading
from correspondences import find_correspondences, draw_correspondences
import csv

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

results = []

lock = threading.Lock()

def load_images(base_path, extension='.png'):  
    images = []  
    for filename in sorted(os.listdir(base_path)):  
        if filename.endswith(extension):  
            img_path = os.path.join(base_path, filename)  
            img = cv2.imread(img_path)  
            if img is not None:  
                images.append(img)  
    return images  
def load_images(base_path, extension='.png'):  
    images = []  
    imgList = os.listdir(base_path)
    imgList.sort(key=lambda x: int(x.split('.')[0]))#按照数字进行排序后按顺序读取文件夹下的图片
    for filename in imgList:  
        if filename.endswith(extension):  
            img_path = os.path.join(base_path, filename)  
            img = cv2.imread(img_path)  
            if img is not None:  
                images.append(img)  
    return images  
  
def process_images(init_path, goal_path, num_pairs, load_size, facet, bin, thresh, model_type, stride):  
    init_images = load_images(init_path)  
    goal_images = load_images(goal_path) 
  
    assert len(init_images) == len(goal_images), "goal and init images must have same numbers"
    
    with open("points.csv", mode='a', newline='') as f:  
        writer = csv.writer(f) 
        writer.writerow(['x1', 'y1', 'x2', 'y2','x3', 'y3', 'x4', 'y4', 'x5', 'y5',
                    'x6', 'y6', 'x7', 'y7','x8', 'y8', 'x9', 'y9', 'x10', 'y10']) 
      
        for i in range(len(init_images)):  
            rgb_1 = init_images[i]  
            rgb_2 = goal_images[i] 
    
            # 调用find_correspondences 
            with torch.no_grad():
                with lock:  
                    points1, points2, image1_pil, image2_pil = find_correspondences(rgb_1, rgb_2, num_pairs, load_size, layer,
                                                                                facet, bin, thresh, model_type, stride)
                
                # fig1, fig2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
                # image_path_1 = f'./keypoints_bn_{i}.png'
                # fig1.savefig(image_path_1)

                # image_path_2 = f'./keypoints_live_{i}.png'
                # fig2.savefig(image_path_2)
                
                # 将 points1   points2  写入到 points.csv 文件中  
                writer.writerow(np.concatenate(points1, axis = 0).tolist())
                writer.writerow(np.concatenate(points2, axis = 0).tolist())

  
# 路径  
init_path = '/home/chen/Desktop/dataset_aa/init_images'  
goal_path = '/home/chen/Desktop/dataset_aa/goal_images'  
  
#Hyperparameters for DINO correspondences extraction
num_pairs = 10  
load_size = 128  # 224
layer = 9  # 0-11
facet = 'key' 
bin=True 
thresh=0.05 
model_type='dino_vits8' 
stride=4  

# 调用函数  
process_images(init_path, goal_path, num_pairs, load_size, facet, bin, thresh, model_type, stride)