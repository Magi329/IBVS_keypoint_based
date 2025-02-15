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

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import threading
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation as R



os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

class Servo(Node):
    def __init__(self, final_pose):
        super().__init__('servoing_node')
        self.bridge = CvBridge()
        # realsense 
        self.fps = 30
        self.process_flag = False
        # image
        self.rgb_goal = None
        self.rgb_live = None

        # xarm
        # self.arm = XArmAPI('192.168.11.11')  # attention
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
        
        # delta_trans[0] *= -0.001
        # delta_trans[1] *= 0.001
        # delta_trans[2] *= 0.001


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
                                
                image_path_1 = f'./keypoints_live_{index}.png'
                fig1.savefig(image_path_1)
                image_path_2 = f'./keypoints_bn_{index}.png'
                fig2.savefig(image_path_2)
                index += 1  
            print("finish corresp")

            points1 = np.array(points1).reshape(1, 2*num_pairs)
            points2 = np.array(points2).reshape(1, 2*num_pairs)

            error = compute_error(points1, points2)
            print("当前误差是:", error)

            # live_pose = self.arm.get_position(is_radian=True) # attention
            live_pose = torch.tensor([-318.34845, 476.89444, 395.504242, 1.614076, -0.791094, -1.838659])
            
            ## 调用模型  ##
            pred_data = self.pred_action(points1, points2, live_pose)
            print("pred_pos", pred_data)          
            
            
#############
#####################################  MOVE XARM  #####################################

            if live_pose[0] + pred_data[0]<-490:# 防碰桌子保护： x>-500mm
                print("ERROR: gripper reaches the desk. STOP\n")
                break

            # self.arm.clean_warn() # attention
            # self.arm.clean_error() # attention

            # 移动机械臂 {相对}距离
            alpha = 0.3


            # self.arm.set_position(x=pred_data[0]*alpha, y=pred_data[1]*alpha, z=pred_data[2]*alpha, roll=0, pitch=0, yaw=0, # 先只测xyz的，录demo出来
            #                      speed=self.speed, mvacc=self.acceleration, relative=True, is_radian=True, wait=True) # attention
            print("----- XARM MOVED -----")

            time.sleep(1)

        print("Error small enough, servoing ends. \n Closing gripper ... \n")
        # self.arm.set_gripper_position(10, speed=6000)  # 需要确定夹取宽度  # attention
        # grasp_pose = [-318.34845, 476.89444, 395.504242, 1.614076, -0.791094, -1.838659] # attention
        # self.arm.set_position(x=grasp_pose[0], y=grasp_pose[1], z=grasp_pose[2], roll=grasp_pose[3], pitch=grasp_pose[4], yaw=grasp_pose[5],
        #                          speed=self.speed, mvacc=self.acceleration, relative=False, is_radian=True, wait=True) # attention
        print("----- SERVOING END -----")



def main(args=None):
    # 启动节点
    rclpy.init()

    final_pose = np.array([0.07246804319526073, 0.15627486690101278, 0.9935891484251596]) # 关闭夹爪时的位姿
    node = Servo(final_pose)                       # attention
    # node.arm.motion_enable(enable=True) # attention
    # node.arm.set_mode(0)                  # attention
    # node.arm.set_state(state=0)          # attention

    # node.arm.set_gripper_enable(True)  # gripper可以用的话把这个注释取消掉
    # node.arm.set_gripper_position(850, speed=6000)# gripper可以用的话把这个注释取消掉
    
    # 获取目标图像
    node.rgb_goal = args.goal_image
    print("Goal image loaded\n")
    rgb_live = 'bag_live.jpg' # attention
    node.rgb_live = rgb_live


    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
        print("spin once")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model checkpoint inference")
    parser.add_argument('--model-rot', type=str, default = '/home/chen/Desktop/checkpoints/rot/0210-7/model.pth')
    parser.add_argument('--model-trans', type=str, default = '/home/chen/Desktop/checkpoints/trans/0213-3/model.pth')
    parser.add_argument('--goal-image', type=str, default = 'bag_goal.jpg')
    args = parser.parse_args()
    main(args)


