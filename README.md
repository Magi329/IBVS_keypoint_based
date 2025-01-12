## 系统的准备：
ubuntu2204

ros2 humble


## 硬件部分的准备：

1.机械臂

xarm文件夹放到和gogo同一个目录，用xarm官方sdk获取机械臂的位姿态、控制机械臂运动。

2.相机

订阅相机的话题，来实时获取腕部相机的图像信息。

下载奥比中光的ROS包，https://github.com/orbbec/OrbbecSDK_ROS2，按照网站的提示下载。

起话题的命令行： ros2 launch orbbec_camera gemini_330_series.launch.py depth_registration:=true


## 软件的准备：
Seg 用于从图片中分割出待抓物体。需要配置环境。文件较大，这里没有上传完整包，需从sam2和florense2官网下载

Dino-vit-features 用于提取图片的关键点。需要配置环境。此包用到了Seg，运行时需修改sys.path.append。需从官网下载https://github.com/ShirAmir/dino-vit-features

xarm 用于获取机械臂的状态、控制机械臂运动。


### 环境配置：
创建python=3.10的conda环境：
conda create -n main python=3.10
### 配置分割：
cd Seg/segment-anything-2 

pip install -e .
### 配置dino：
 conda install pytorch torchvision torchaudio cudatoolkit=11 -c pytorch
 
 conda install tqdm
 
 conda install -c conda-forge faiss

 conda install -c conda-forge timm 
 
 conda install matplotlib
 
 pip install opencv-python

 pip install git+https://github.com/lucasb-eyer/pydensecrf.git

 conda install -c anaconda scikit-learn

（其实就是按照官网的提示输入命令行）




## TroubleShooting：
conda环境用ROS2，rclpy importError参考https://blog.csdn.net/weixin_44506963/article/details/145101457


