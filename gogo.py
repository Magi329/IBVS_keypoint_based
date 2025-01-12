import torch
import torchvision.transforms as T
import warnings 
import time
warnings.filterwarnings("ignore")
import pyrealsense2 as rs
import sys
sys.path.append('./dino-vit-features')
from correspondences import find_correspondences, draw_correspondences  
from my_utils.MODEL_v2_2 import QuaternionAlignmentTransformer, Alpha
from my_utils.helper_functions import euler_to_quaternion, quaternion_to_euler, compute_error
import my_utils.quaternion_calc as qc
from my_utils.parameters import *

from xarm.wrapper import XArmAPI
import os
import itertools

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import threading
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation as R



os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

class Args(Node):
    def __init__(self):
        super().__init__('servoing_node')
        self.bridge = CvBridge()
        # realsense 
        self.fps = 30
        self.process_flag = False
        # image
        self.rgb_goal = None
        self.rgb_live = None

        # xarm
        #a self.arm = XArmAPI('192.168.11.11')
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
        self.model_trans.load_state_dict(torch.load('/home/chen/Desktop/checkpoints/weight_test/svd_bag_1219_5_799.pth'))
        self.model_trans.eval()
        self.model_rot = QuaternionAlignmentTransformer(hidden_dim, hidden_depth_dim, num_heads, num_layers, input_dim)
        self.model_rot.load_state_dict(torch.load('/home/chen/Desktop/checkpoints/frozeTrans/svd_bag_1219_transformer_v2.2.pth'))
        self.model_rot.eval()

        self.lock = threading.Lock()
        # servoing thread
        self.servo_thread = threading.Thread(target = self.servoing)
        self.servo_thread.start()
        

    def color_callback(self, msg):
        self.rgb_live = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')


    def calib_camera_ext(self, R):   
        return np.dot(self.R_cam2gripper, R)
    

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
            print("current error is:", error)

            #a live_pose = self.arm.get_position(is_radian=True)
            live_pose = torch.tensor([0,0,0,0,0,0,0])

############
############################# 调用模型  ##################################
            points1 = torch.tensor(list(itertools.chain(*points1)), dtype=torch.int)
            points2 = torch.tensor(list(itertools.chain(*points2)), dtype=torch.int)
            
            x = torch.cat((points1, points2), dim=0).float()

            # output of tanslation :
            delta_trans = self.model_trans(x)[:3]
            predicted_trans = live_pose[:3] + delta_trans
            depth = torch.norm(torch.tensor([0, 0, 15.0]), dim=-1).unsqueeze(-1)   # of goal pose

            # output of rotation
            live_pose_quat = torch.tensor(euler_to_quaternion(live_pose))
            output_quaternion = self.model_rot(live_pose_quat.to(torch.float32), x, delta_trans, depth.to(torch.float32)).detach().numpy() 
            predicted_quaternion = qc.batch_concatenate_quaternions(torch.tensor(live_pose_quat[3:]), torch.tensor(output_quaternion, dtype=float))
            
            # final predicted eef pose
            pred_pose_quat = np.concatenate((predicted_trans.detach().numpy() , predicted_quaternion), axis=0)
            pred_pose = quaternion_to_euler(pred_pose_quat)
            
            if mode == 'quaternion':
                pred_data = pred_pose_quat

            elif mode == 'absolute':
                pred_data = pred_pose
            
            elif mode == 'relative':
                pred_rot = pred_pose[3:]
                # pred_rot = (pred_rot + np.pi) % (2 * np.pi) - np.pi
                pred_delta_pose = np.concatenate((delta_trans.detach().numpy(), pred_rot), axis=0)
                pred_data = pred_delta_pose          
            
            
#############
#####################################  MOVE XARM  #####################################
            
            # print the predicted pose
            print("pred_pos", pred_data) 

            if pred_data[0]<-490:# 防碰桌子保护： x>-500mm
                print("ERROR: gripper reaches the desk. STOP\n")
                break

            #a self.arm.clean_warn()
            #a self.arm.clean_error()

            # 移动机械臂 {相对}距离
            alpha = 0.3

            # self.arm.set_servo_angle(angle=predict_poses[0], relative=True, is_radian=True, wait=True)
            #a self.arm.set_position(x=pred_data[0]*alpha, y=pred_data[1]*alpha, z=pred_data[2]*alpha, roll=pred_data[3], pitch=pred_data[4], yaw=pred_data[5],
            #a                      speed=self.speed, mvacc=self.acceleration, relative=False, is_radian=True, wait=True)
            print("----- XARM MOVED -----")

            time.sleep(1)

        print("Error small enough, servoing ends. \n Closing gripper ... \n")
        #a self.arm.set_gripper_position(10, speed=6000)  # 需要确定夹取宽度
        final_pose = [-318.34845, 476.89444, 395.504242, 1.614076, -0.791094, -1.838659]
        #a self.arm.set_position(x=final_pose[0], y=final_pose[1], z=final_pose[2], roll=[3], pitch=[4], yaw=[5],
        #a                          speed=self.speed, mvacc=self.acceleration, relative=False, is_radian=True, wait=True)
        print("----- SERVOING END -----")





def main(args=None):
    # 启动节点
    rclpy.init(args=args)
    node = Args()
    #a node.arm.motion_enable(enable=True)
    #a node.arm.set_mode(0)
    #a node.arm.set_state(state=0)
    # node.arm.set_gripper_enable(True)
    # node.arm.set_gripper_position(850, speed=6000)
    # 获取目标图像
    rgb_goal = 'bag_goal.jpg'
    node.rgb_goal = rgb_goal
    print("Goal image loaded\n")
    rgb_live = 'bag_live.jpg'
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
    main()


