import torch  
import torch.nn as nn  
import numpy as np  
from scipy.spatial.transform import Rotation as R

# 定义模型结构，与训练时相同  
class MLP(nn.Module):  
    def __init__(self):  
        super(MLP, self).__init__()  
        self.fc1 = nn.Linear(28, 64)  
        self.fc2 = nn.Linear(64, 128)  
        self.fc3 = nn.Linear(128, 64)  
        self.fc4 = nn.Linear(64, 7)  
  
    def forward(self, x):  
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x))  
        x = torch.relu(self.fc3(x))  
        x = self.fc4(x)  
        return x  
   
def load_model(model_path):  
    model = MLP()  
    model.load_state_dict(torch.load(model_path))  
    model.eval()  # 设置为评估模式  
    return model  
   
def predict(model, inputs):  
    # 确保输入是torch张量，并且是在正确的设备上（CPU或GPU）  
    inputs = torch.tensor(inputs, dtype=torch.float32)  
    with torch.no_grad():    
        predictions = model(inputs)  
    result = predictions.numpy()
    return result

def quaternion_to_euler(pose):
    """
    将pose后四位的四元数 (x, y, z, w) 转换为欧拉角 (roll, pitch, yaw),再并到pose里
    
    参数:
    quaternion -- 四元数 [x, y, z, w]
    
    返回:
    欧拉角 [roll, pitch, yaw] (弧度)
    """
    # 使用 scipy 的 Rotation 类将四元数转换为欧拉角
    rotation = R.from_quat(pose[3:])
    euler_angles = rotation.as_euler('xyz')  # 返回格式为 [roll, pitch, yaw]
    
    return np.concatenate((np.array(pose[3:]), euler_angles))
 
def quaternion_to_euler(pose):
    """
    将pose后四位的四元数 (x, y, z, w) 转换为欧拉角 (roll, pitch, yaw),再并到pose里
    
    参数:
    quaternion -- 四元数 [x, y, z, w]
    
    返回:
    欧拉角 [roll, pitch, yaw] (弧度)
    """
    rotation = R.from_quat(pose[3:7])
    euler_angles = rotation.as_euler('xyz')  # 返回格式为 [roll, pitch, yaw]
    
    return np.concatenate((np.array(pose[:3]), euler_angles))
  
# def main():   
#     # pose = [1,2,3,4,5,6,7]
#     # res = quaternion_to_euler(pose)
#     # print(res)
#     model_path = 'pengfei_test_v1.pth'  
#     model = load_model(model_path)  
  
#     # 示例输入数据，假设是40个特征（例如两个图像的20个关键点坐标）   
#     example_input = np.array([88, 224, 40, 136, 32, 220, 76, 208, 84, 200, 68, 216, 76, 296, 
#                             112, 188, 76, 100, 44, 156, 88, 168, 104, 160, 80, 188, 56, 320])  # 假设我们有一个样本  
  
#     predictions = predict(model, example_input)  
  
#     print("Predicted pose:", predictions)  
  
# if __name__ == "__main__":  
#     main()
































