import numpy as np

#Hyperparameters for DINO correspondences extraction
num_pairs = 7
load_size = 224
layer = 9  # 0-11
facet = 'key' 
bin=True 
thresh=0.05 
model_type='dino_vits8' 
stride=4 

#Deployment hyperparameters    
ERR_THRESHOLD = 0.1  #A generic error between the two sets of points

# camera resolution
resolution = (640, 480)
# color相机内参
fx_color = 908.4378662109375
fy_color = 908.0997314453125
cx_color = 648.5411376953125
cy_color = 380.6193542480469
K_color = np.array([
    [fx_color, 0, cx_color],
    [0, fy_color, cy_color],
    [0, 0, 1]
])
color_width = 1280
color_height = 720

# 深度相机内参
fx_depth = 428.28594970703125
fy_depth = 428.28594970703125
cx_depth = 428.1567687988281
cy_depth = 242.36892700195312
K_depth = np.array([
    [fx_depth, 0, cx_depth],
    [0, fy_depth, cy_depth],
    [0, 0, 1]
])
depth_width = 848
depth_height = 480 
