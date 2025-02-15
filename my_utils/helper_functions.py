import numpy as np
import PIL.Image as PILimage
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('/home/chen/Packages/Seg/florence_sam')
from florence_sam import Segmentation   # 加mask的
from my_utils.parameters import PROMPT
import torch


def pose_euler_to_quaternion(pose):
    """
    将pose后三项欧拉角 (roll, pitch, yaw) 转换为四元数 (x, y, z, w), 再并回pose里

    返回: [xx, yy, zz, x, y, z, w]
    """
    if isinstance(pose, str):
        try:
            # Attempt to convert the string to a list of floats
            pose = [float(num) for num in pose.strip('[]').split(', ')]
        except ValueError as e:
            # Handle the error if parsing fails
            print(f"Error parsing pose string: {e}")
            return None
    rotation = R.from_euler('xyz', [pose[3], pose[4], pose[5]], degrees=False)
    quaternion = rotation.as_quat()  # 返回格式为 [x, y, z, w]
    
    return torch.concatenate((torch.tensor(pose[:3]), torch.tensor(quaternion)), dim=0).view(-1)



def quaternion_to_euler(pose):
    """
    将pose后四位的四元数 (x, y, z, w) 转换为欧拉角 (roll, pitch, yaw),再并到pose里
    
    参数:
    quaternion -- 四元数 [x, y, z, w]
    
    返回:
    欧拉角 [roll, pitch, yaw] (弧度)
    """
    rotation = R.from_quat(pose)
    euler_angles = rotation.as_euler('xyz')  # 返回格式为 [roll, pitch, yaw]
    
    return euler_angles #np.concatenate((np.array(pose[:3]), euler_angles))


def compute_error(points1, points2):
    return np.linalg.norm(np.array(points1) - np.array(points2))


def isRotationMatrix(R) :
    '''Checks if a matrix is a valid rotation matrix.'''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 

def add_mask(input_image):
    '''把input_image的mask以外的部分设置成黑色 '''
    input_image = PILimage.fromarray(input_image)
    if 'Seg' not in locals():
        Seg = Segmentation()
    _, mask = Seg.process_image(input_image, PROMPT)
    input_image_np = np.array(input_image)
    mask_np = np.array(mask[0]).astype(bool)

    black_image = np.zeros_like(input_image_np)
    black_image[mask_np] = input_image_np[mask_np]
    result_image = PILimage.fromarray(black_image)

    return result_image