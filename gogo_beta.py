import torch
import torchvision.transforms as T
import warnings 
import time
warnings.filterwarnings("ignore")
import sys
sys.path.append('./dino-vit-features')
from correspondences import find_correspondences, draw_correspondences  
from my_utils.MODEL_v2_2 import QuaternionAlignmentTransformer, Alpha
from my_utils.helper_functions import pose_euler_to_quaternion, quaternion_to_euler, compute_error
import my_utils.quaternion_calc as qc
from my_utils.parameters import *

from xarm.wrapper import XArmAPI
import os
import itertools
import argparse
import logging

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import threading
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation as R



os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

class Servo(Node):
    def __init__(self, final_pose, weight, logger):
        super().__init__('servoing_node')
        self.bridge = CvBridge()
        # realsense 
        self.fps = 30
        self.process_flag = False
        # image
        self.rgb_goal = None
        self.rgb_live = None

        # xarm
        if args.real_world == True:
            self.arm = XArmAPI('192.168.11.11') 
            self.speed = 30 #50 
            self.acceleration = 30 #50 
        # camera extrinsics
        self.R_cam2gripper = np.eye(3)

        # Create subscriptions
        self.color_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_callback,
            10)
        self.color_sub

        # 加载模型
        self.model_trans = Alpha()
        self.model_trans.load_state_dict(torch.load(args.model_trans))
        self.model_trans.eval()
        self.model_rot = QuaternionAlignmentTransformer(hidden_dim, hidden_depth_dim, num_heads, num_layers, input_dim)
        self.model_rot.load_state_dict(torch.load(args.model_rot))
        self.model_rot.eval()

        self.final_pose = final_pose
        self.weight = weight
        self.logger = logger

        self.lock = threading.Lock()
        # servoing thread
        self.servo_thread = threading.Thread(target = self.servoing)
        self.servo_thread.start()
        

    def color_callback(self, msg):
        self.rgb_live = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')


    def calib_camera_ext(self, R):   
        return np.dot(self.R_cam2gripper, R)
    
    @staticmethod
    def _delta_euler_from_quat(curr_quat, goal_quat):
        """
        计算两个四元数之间的欧拉角差值，并解决相位跳变问题。

        参数:
        curr_quat -- 当前四元数 [x, y, z, w]
        goal_quat -- 目标四元数 [x, y, z, w]

        返回:
        delta -- 欧拉角差值 [roll, pitch, yaw] (弧度)，仅保留最大差值维度，其余为0
        """
        # 将四元数转换为欧拉角
        curr_euler = quaternion_to_euler(np.array(curr_quat))
        goal_euler = quaternion_to_euler(np.array(goal_quat))

        # 计算差值
        delta = goal_euler - curr_euler

        # 解决相位跳变问题（将差值限制在 [-pi, pi] 范围内）
        for i in range(3):
            if delta[i] > np.pi:
                delta[i] -= 2 * np.pi
            elif delta[i] < -np.pi:
                delta[i] += 2 * np.pi

        # 找到最大差值的维度
        max_index = np.argmax(np.abs(delta))
        max_delta = delta[max_index]

        # 仅保留最大差值维度，其余维度设为0
        delta = np.zeros_like(delta)
        delta[max_index] = max_delta

        return delta
    
    def pred_action(self, points1, points2, live_pose):
        '''预测eef位姿变换'''
        points1 = torch.tensor(list(itertools.chain(*points1)), dtype=torch.int)
        points2 = torch.tensor(list(itertools.chain(*points2)), dtype=torch.int)
        
        x = torch.cat((points1, points2), dim=0).float()

        # output of tanslation :
        delta_trans = self.model_trans(x)[:3]

        depth = torch.norm((live_pose[:3] - torch.tensor(self.final_pose)), 
                           dim=-1).unsqueeze(-1)

        # output of rotation
        live_pose_quat = pose_euler_to_quaternion(live_pose)
        output_quaternion = self.model_rot(live_pose_quat.float(), 
                                           x, 
                                           delta_trans, 
                                           depth.float()).detach().numpy() 
        
        predicted_quaternion = qc.batch_concatenate_quaternions(live_pose_quat[3:], 
                                                                torch.tensor(output_quaternion, 
                                                                dtype=float))
        
        pred_delta_pose = np.concatenate((delta_trans.detach().numpy(), 
                                          self._delta_euler_from_quat(live_pose_quat[3:], predicted_quaternion.detach().numpy())), 
                                          axis=0)
        
        return pred_delta_pose



    def servoing(self):
        # visual servoing 对齐阶段
        print("已拍摄目标图像, 等待3秒...")
        time.sleep(3)  
        print("开始执行视觉伺服...")


        error = 100000
        index = 0
        while error > ERR_THRESHOLD:

            #Compute pixel correspondences between new observation and bottleneck observation.
            with torch.no_grad():
                with self.lock:
                    points1, points2, image1_pil, image2_pil = find_correspondences(PROMPT, self.rgb_live, self.rgb_goal, num_pairs, load_size, layer,
                                                                                facet, bin, thresh, model_type, stride)
            
                fig1, fig2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
                                
                image_path_1 = os.path.join(args.log_dir, f'./keypoints_live_{index}.png') 
                fig1.savefig(image_path_1)
                image_path_2 = os.path.join(args.log_dir, f'./keypoints_goal_{index}.png') 
                fig2.savefig(image_path_2)
                index += 1  
            print("finish corresp")

            points1 = np.array(points1).reshape(1, 2*num_pairs)
            points2 = np.array(points2).reshape(1, 2*num_pairs)

            error = compute_error(points1, points2)
            self.logger.info("当前误差是: %s", error)

            if args.real_world == True:
                live_pose = self.arm.get_position(is_radian=True) 
            else:
                live_pose = torch.tensor([-318.34845, 476.89444, 395.504242, 1.614076, -0.791094, -1.838659])
            
            ## 调用模型  ##
            prim_command = self.pred_action(points1, points2, live_pose)
            self.logger.info("weighted command: %s", prim_command)

            command = [cmd * weight for cmd, weight in zip(prim_command, self.weight)]  
            self.logger.info("weighted command %s", prim_command)      
            
            
            ########  MOVE XARM  #######
            if live_pose[0] + command[0]<-490:# 防碰桌子保护： x>-500mm
                self.logger.info("ERROR: gripper reaches the desk. STOP\n")
                break
            if args.real_world == True:
                self.arm.clean_warn() 
                self.arm.clean_error() 
                self.arm.set_position(x=command[0], y=command[1], z=command[2], roll=command[3], pitch=command[4], yaw=command[5], 
                                     speed=self.speed, mvacc=self.acceleration, relative=True, is_radian=True, wait=True) 
            self.logger.info("----- XARM MOVED -----")

            time.sleep(1)

        self.logger.info("Error small enough, servoing ends. \n Closing gripper ... \n")
        
        self.logger.info("----- SERVOING END -----")


## 配置logger
def configure_logger(log_dir: str) -> logging.Logger:
    """Configure and return training logger.
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'servo.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main(args=None):
    # 启动节点
    rclpy.init() 
    # 启动logger
    logger = configure_logger(args.log_dir)

    # 抓到物体时的位姿
    final_pose = np.array([0.07246804319526073, 0.15627486690101278, 0.9935891484251596]) # 关闭夹爪时的3D位姿
    
    # 在 x y z roll pitch yaw 方向上的运动方向、幅度控制
    weight = [1, 1, 1, 1, 1, 1] 

    node = Servo(final_pose, weight, logger) 

    if args.real_world == True:                      
        node.arm.motion_enable(enable=True) 
        node.arm.set_mode(0)                  
        node.arm.set_state(state=0)          

        # node.arm.set_gripper_enable(True)  # gripper可以用的话把这个注释取消掉
        # node.arm.set_gripper_position(850, speed=6000)# gripper可以用的话把这个注释取消掉
    
    # 获取目标图像
    node.rgb_goal = args.goal_image
    print("Goal image loaded\n")

    if args.real_world == False:
        rgb_live = 'bag_live.jpg' 
        node.rgb_live = rgb_live


    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model checkpoint inference")
    parser.add_argument('--model-rot', type=str, 
                        default = '/home/chen/Desktop/checkpoints/rot/0210-7/model.pth')
    parser.add_argument('--model-trans', type=str, 
                        default = '/home/chen/Desktop/checkpoints/trans/0213-3/model.pth')
    parser.add_argument('--goal-image', type=str, 
                        default = './bag_goal.jpg')
    parser.add_argument('--real-world', type=bool, 
                        default = False)
    parser.add_argument("--log-dir", type=str, 
        # default='./servo_log_0215-1',  # 默认日志保存路径
        help="Directory for saving logs and images"
    )
    args = parser.parse_args()
    main(args)


