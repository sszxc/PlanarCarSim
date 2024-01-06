# Author: Xuechao Zhang
# Date: Feb 13th, 2022
# Description: 虚拟相机：给定地面纹理和相机位姿，输出图像

import cv2
import numpy as np
import random
import math
import copy
import time
import yaml
import sys
from src.distortion_para_fit import *
from src.utils import *
from src.car_controller import CarController

class VirtualCamEnv:
    def __init__(self, config_file_path, background_path, borderValue=(0, 0, 0), AprilTag_detection=False):
        '''
        多相机虚拟环境

        config_file_path: 相机配置文件路径
        background_path: 背景图片路径
        AprilTag_detection: 是否进行 AprilTag 检测任务
        '''
        # 所有相机列表
        all_cameras=[]
        # 从config文件中读取参数 创建虚拟相机
        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)
            for camera_name, value in config.items():
                all_cameras.append(Cam(camera_name, value, AprilTag_detection))

        # 当前相机序号
        self.index = 0
        # 理想俯视图相机
        try:
            self.topview = [cam for cam in all_cameras if cam.name == 'topview'][0]
        except IndexError:
            print("We need a topview camera. Please check the config file.")
            exit(1)
        # 普通节点相机
        self.cameras = [cam for cam in all_cameras if cam.name != 'topview']
        assert len(self.cameras) > 0, "We need at least one camera. Please check the config file."

        # 设定平面上四个参考点
        self.reference_points = set_reference_points()
        self.points_topview = [self.topview.world_point_to_cam_pixel(point) for point in self.reference_points[0:4]]

        # 导入背景图 调整比例
        background_img = cv2.imread(background_path)
        source_h, source_w, _ = background_img.shape
        target_h, target_w, _ = self.topview.img.shape
        ratio = min(target_h/source_h, target_w/source_w)
        background_img = cv2.resize(background_img, None, fx=ratio, fy=ratio)
        # 不匹配的比例用白色填充
        source_h, source_w, _ = background_img.shape
        if source_h != target_h or source_w != target_w:
            cprint("The background image should have the same aspect ratio as the topview camera resolution. We will fill the unmatched area with white color.",
                   "yellow")
            if source_h < target_h:
                border = (int((target_h-source_h)/2), int((target_h-source_h)/2), 0, 0)
            else:
                border = (0, 0, int((target_w-source_w)/2), int((target_w-source_w)/2))
            background_img = cv2.copyMakeBorder(background_img, *border, cv2.BORDER_CONSTANT, value=borderValue)
            # cv2.BORDER_REPLICATE  # 复制法填充
        self.background = background_img

        # assert len(self.cameras) > 0, "We need at least one camera. Please check the config file."

        print("Environment set!")

    def render(self):
        '''渲染相机图像'''
        # 手动更新俯视图相机的图像
        self.topview.img = copy.deepcopy(self.background)
        
        for cam in self.cameras:
            # 利用前四个点生成变换矩阵
            cam.update_M(self.reference_points[0:4], self.points_topview)

            # 更新本相机图像
            cam.update_img(self.background)

            # 更新俯视图外框
            cam.cam_frame_project(self.topview, (140, 144, 32))

        return [(cam.img, cam.name) for cam in self.cameras] + [(self.topview.img, 'topview')]

    def control(self, T_para, command, real_kinematics=True):
        '''更新小车位姿'''
        if real_kinematics:
            if 'car_controller' not in globals():
                global car_controller
                car_controller = CarController()
            car_controller.read_command(T_para)
            self.cameras[self.index].update_car_pose(
                *list_add(car_controller.get_dT(), self.cameras[self.index].car_para))
        else:
            self.cameras[self.index].update_car_pose(
                *list_add(T_para, self.cameras[self.index].car_para))  # 更新外参
        
        if command == -1:  # 解析其他命令
            sys.exit()
        elif command == 1:
            self.cameras[self.index].reset_car_pose()
        elif command == 2:
            self.index = (self.index + 1) % len(self.cameras)
        elif command == 3:
            self.index = (self.index - 1) % len(self.cameras)
        
        return self.cameras[self.index].car_para, self.cameras[self.index].cam_para, "cam" + str(self.index)

    def __del__(self):
        '''析构函数'''
        print("Environment closed!")        


class Cam:
    def __init__(self, camera_name, parameters, AprilTag_detection):
        '''
        由相机代表的小车对象
        相机实际外参 = self.car_pose * self.EM
        '''
        # 初始化内外参
        self.name = camera_name
        self.expand_for_distortion = 0.15  # 每侧扩张比例
        self.update_IM(*parameters["intrinsic parameters"])
        self.update_EM(*parameters["extrinsic parameters"])
        self.update_car_pose(*parameters["initial pose"])
        self.car_pose_default = copy.deepcopy(parameters["initial pose"])  # 存储初始值

        self.height, self.width = parameters["resolution"]
        self.init_view(*parameters["resolution"])
        self.distortion = parameters["distortion"]

        if self.distortion:  # 畸变模拟
            self.init_distortion_para(self.distortion)

        self.AprilTag_detection = AprilTag_detection  # AprilTag 检测任务
        if self.AprilTag_detection:
            from src.apriltag_utils import ApriltagDetector
            self.AprilTag = ApriltagDetector(parameters["intrinsic parameters"])
        
        print(f"Camera {self.name} set!")

    def update_IM(self, fx, fy, cx, cy):
        '''初始化相机内参'''
        self.IM = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]],
            dtype=float
        )

    def Euler_to_RM(self, theta):
        '''从欧拉角到旋转矩阵'''
        theta = [x / 180.0 * 3.14159265 for x in theta]  # 角度转弧度
        R_x = np.array([[1,                  0,                   0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0,  math.sin(theta[0]), math.cos(theta[0])]])
        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0,                  1,                  0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]])
        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]),  0],
                        [0,                  0,                   1]])
        # R = np.dot(R_y, np.dot(R_x, R_z))
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def EulerXYZ_to_RM(self, euler, dx, dy, dz):
        RM = self.Euler_to_RM(euler)
        T_43 = np.r_[RM, [[0, 0, 0]]]
        T_rotate = np.c_[T_43, [0, 0, 0, 1]]
        T_trans = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]],
            dtype=float
        )
        return np.dot(T_rotate, T_trans)

    def RM_to_EulerXYZ(self, RM):
        '''从旋转矩阵到欧拉角'''
        sy = math.sqrt(RM[0, 0] * RM[0, 0] + RM[1, 0] * RM[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(RM[2, 1], RM[2, 2])
            y = math.atan2(-RM[2, 0], sy)
            z = math.atan2(RM[1, 0], RM[0, 0])
        else:
            x = math.atan2(-RM[1, 2], RM[1, 1])
            y = math.atan2(-RM[2, 0], sy)
            z = 0
        Euler = np.array([x, y, z]) * 180.0 / np.pi  # 转换为角度
        # return Euler, RM[0, 3], RM[1, 3], RM[2, 3]
        X, Y, Z = -np.linalg.inv(RM)[0:3, 3]
        return Euler, X, Y, Z

    def update_EM(self, euler, dx, dy, dz):
        self.EM = self.EulerXYZ_to_RM(euler, dx, dy, dz)

    def update_car_pose(self, euler, dx, dy, dz):
        '''
        更新外参 即相机与世界之间的转换矩阵
        更新相机中轴线与地面交点
        '''
        self.car_para = [euler, dx, dy, dz]
        T44_car_to_world = self.EulerXYZ_to_RM(euler, dx, dy, dz)
        T44_cam_to_car = self.EM
        self.T44_cam_to_world = np.dot(T44_cam_to_car, T44_car_to_world)
        self.T44_world_to_cam = np.linalg.inv(self.T44_cam_to_world)

        self.cam_para = self.RM_to_EulerXYZ(self.T44_cam_to_world)
        self.dz = self.cam_para[-1]

        self.update_central_axis_cross_ground()  # 更新光轴交点

    def update_central_axis_cross_ground(self):
        '''计算光轴与地面交点'''
        euler, dx, dy, dz = self.cam_para
        # 竖直向下的向量经过RM矩阵旋转得到法向
        direction = np.dot(
            np.array([0, 0, 1], dtype=np.float), self.Euler_to_RM(euler))  # 方向向量
        focus_x = dx - dz/direction[2]*direction[0]
        focus_y = dy - dz/direction[2]*direction[1]
        self.central_axis_cross_ground = np.array([[-focus_x], [-focus_y], [0]], dtype=float)
        # self.central_axis_cross_ground = np.array([[-dx], [-dy], [0]], dtype=float)

    def reset_car_pose(self):
        '''小车外参归位'''
        self.update_car_pose(*self.car_pose_default)

    def update_M(self, points_3D, points_topview):
        '''
        用几组点对计算相机到全局的图片变换矩阵
        '''
        pixels = [self.world_point_to_cam_pixel(point) for point in points_3D]
        # 为畸变模拟服务 目标位置平移
        if self.distortion:
            pixels_expand = []
            for i in pixels:
                pixels_expand.append((i[0] + self.expand_for_distortion*self.height,
                                i[1] + self.expand_for_distortion*self.width))
            self.M33_global_to_local_expand = cv2.getPerspectiveTransform(np.float32(points_topview), 
                                                                    np.float32(pixels_expand))
        # 非畸变情况
        self.M33_global_to_local = cv2.getPerspectiveTransform(np.float32(points_topview), 
                                                                np.float32(pixels))
        # 注意这里畸变情况下也用标准的 local_to_global 仅在投影到俯视图中会调用
        self.M33_local_to_global = np.linalg.inv(self.M33_global_to_local)

    def init_view(self, height, width, color=(255, 255, 255)):
        '''
        新建图像，并设置分辨率、背景色
        '''
        graph = np.zeros((height, width, 3), np.uint8)
        graph[:, :, 0] += color[0]
        graph[:, :, 1] += color[1]
        graph[:, :, 2] += color[2]
        self.img = graph

    def world_point_to_cam_point(self, point):
        '''
        从世界坐标点到相机坐标点
        补上第四维的齐次 1, 矩阵乘法, 去掉齐次 1
        '''
        point_in_cam = np.dot(self.T44_cam_to_world, np.r_[point, [[1]]])
        return np.delete(point_in_cam, 3, 0)

    def world_point_to_cam_pixel(self, point):
        '''
        从世界坐标点到像素坐标点 → “拍照”
        '''
        point_in_cam = self.world_point_to_cam_point(point)
        dot = np.dot(self.IM, point_in_cam)
        dot /= dot[2]  # 归一化
        pixel = tuple(dot.astype(np.int).T.tolist()[0][0:2])
        return pixel

    def cam_frame_project(self, topview, color):
        '''
        根据俯视相机参数 把相机视野轮廓投影到俯视图
        '''
        # 标记小车位置 (背景图上 car_para 位置叠加一张 png 图片)
        car_center = self.car_para[1:3]
        car_direction = self.car_para[0][2]
        car_img = cv2.imread("img/car.png", cv2.IMREAD_UNCHANGED)
        target_size = min(topview.img.shape[0], topview.img.shape[1])//10
        car_img = cv2.resize(car_img, (target_size, target_size))
        # 旋转指定角度
        M = cv2.getRotationMatrix2D((target_size//2, target_size//2), car_direction, 1)
        car_img = cv2.warpAffine(car_img, M, (target_size, target_size))
        # 贴到背景图上
        car_center_projected = topview.world_point_to_cam_pixel(np.array([[-car_center[0]], [-car_center[1]], [0]]))
        car_center_projected = (car_center_projected[0]-target_size//2, car_center_projected[1]-target_size//2)
        topview.img[car_center_projected[1]:car_center_projected[1]+target_size,
                    car_center_projected[0]:car_center_projected[0]+target_size] = \
            car_img[:, :, 0:3] * (car_img[:, :, [-1]]/255) + \
            topview.img[car_center_projected[1]:car_center_projected[1]+target_size,
                        car_center_projected[0]:car_center_projected[0]+target_size] * \
            (1 - car_img[:, :, [-1]]/255)

        # 节点相机的角点
        corners = []
        if self.distortion:  # 畸变情况下 由于画面变化 轮廓角点也要调整
            scale = self.expand_for_distortion
            corners.append(np.array([scale*self.width, scale*self.height, 1], dtype=np.float32))
            corners.append(np.array([(scale+1)*self.width, scale*self.height, 1], dtype=np.float32))
            corners.append(np.array([scale*self.width, (scale+1)*self.height, 1], dtype=np.float32))
            corners.append(np.array([(scale+1)*self.width, (scale+1)*self.height, 1], dtype=np.float32))
        else:
            corners.append(np.array([0, 0, 1], dtype=np.float32))
            corners.append(np.array([self.width, 0, 1], dtype=np.float32))
            corners.append(np.array([0, self.height, 1], dtype=np.float32))
            corners.append(np.array([self.width, self.height, 1], dtype=np.float32))
        # 从相机角点到投影图
        projected_corners = []
        for corner in corners:
            projected_corner = np.dot(self.M33_local_to_global, corner.T)
            projected_corner /= projected_corner[2]  # 归一化
            projected_corners.append(tuple(projected_corner.astype(
                np.int).T.tolist()[0:2]))
            cv2.circle(topview.img, tuple(projected_corner.astype(
                np.int).T.tolist()[0:2]), 10, color, -1)
        cv2.line(topview.img, projected_corners[0], projected_corners[1], color,5)
        cv2.line(topview.img, projected_corners[1], projected_corners[3], color,5)
        cv2.line(topview.img, projected_corners[2], projected_corners[3], color,5)
        cv2.line(topview.img, projected_corners[2], projected_corners[0], color,5)
        
        # 标记相机视角中心
        focus_center = topview.world_point_to_cam_pixel(self.central_axis_cross_ground)
        color_for_grid = (130, 57, 68)
        color_for_testbed = (0, 255, 255)
        cv2.circle(topview.img, focus_center, 10, color_for_grid, -1)
        
        # 标记相机固定中心的投影
        camera_center_projected = topview.world_point_to_cam_pixel(np.array([[-self.cam_para[1]], [-self.cam_para[2]], [0]]))
        cv2.line(topview.img, projected_corners[0], camera_center_projected, color,2)
        cv2.line(topview.img, projected_corners[1], camera_center_projected, color,2)
        cv2.line(topview.img, projected_corners[2], camera_center_projected, color,2)
        cv2.line(topview.img, projected_corners[3], camera_center_projected, color,2)
        color_for_grid = (69, 214, 144)
        cv2.circle(topview.img, camera_center_projected, 10, color_for_grid, -1)

    def init_distortion_para(self, distortion_parameters):
        '''
        正向模拟畸变
        '''
    
        data_x, data_y = generate_data(distortion_func_3para, distortion_parameters)  # 真实数据

        popt, pcov = para_fit(data_x, data_y, distortion_func_6para)

        k1, k2, k3, k4, k5, k6 = popt
        p1, p2 = 0, 0
        k = self.IM

        scale = 2*self.expand_for_distortion+1
        k[0,2]*=scale
        k[1,2]*=scale
        d = np.array([
            k1, k2, p1, p2, k3, k4, k5, k6
        ])
        self.distortion_mapx, self.distortion_mapy = cv2.initUndistortRectifyMap(
            k, d, None, k, (int(self.width*scale), int(self.height*scale)), 5)

    def update_img(self, background, borderValue=(0, 0, 0), DEBUG=0):
        '''
        更新图像 同时实现可选的畸变
        borderValue: 控制边界填充颜色
        '''
        if self.distortion:
            # 扩展画布
            scale = self.expand_for_distortion*2+1
            self.img_expand = cv2.warpPerspective(
                background, self.M33_global_to_local,
                (int(self.width*scale), int(self.height*scale)),
                borderValue=borderValue)
            # 正向模拟畸变
            self.img_expand = cv2.remap(self.img_expand, 
                self.distortion_mapx, self.distortion_mapy, 
                cv2.INTER_LINEAR)
            # 导出没有扩张视野的图像
            if DEBUG:
                # 调试看全貌
                self.img = cv2.resize(self.img_expand, (self.img.shape[1], self.img.shape[0]))
            else:
                y_0=int(self.expand_for_distortion*self.height)
                y_1=int((1+self.expand_for_distortion)*self.height)
                x_0=int(self.expand_for_distortion*self.width)
                x_1=int((1+self.expand_for_distortion)*self.width)
                self.img = self.img_expand[y_0:y_1, x_0:x_1]
        else:
            self.img = cv2.warpPerspective(
                background, self.M33_global_to_local, 
                (self.width, self.height), 
                borderValue=borderValue)
        
        # 进行 AprilTag 检测
        if self.AprilTag_detection:
            tags = self.AprilTag.detect(self.img, self.T44_world_to_cam)
            self.AprilTag.draw(self.img, tags)

def list_add(a, b):
    '''
    一种将list各个元素相加的方法
    '''
    c = []
    for i, j in zip(a,b):
        if not isinstance(i,list):  # 如果不是list那直接相加
            c.append(i+j)
        else:  # 否则递归
            c.append(list_add(i,j))
    return c

def randomColor():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def set_reference_points():    
    '''
    在地面上设置四个定位空间点
    '''
    points = [np.array([[0], [0], [0]], dtype=float)]*4
    points[0] = np.array([[0], [0], [0]], dtype=float)
    points[1] = np.array([[0], [3000], [0]], dtype=float)
    points[2] = np.array([[5000], [3000], [0]], dtype=float)
    points[3] = np.array([[5000], [0], [0]], dtype=float)
    return points

def undistort(frame):
    '''
    畸变矫正
    '''
    if 'mapx' not in globals():
        global mapx, mapy
        # 一组去畸变参数
        fx = 827.678512401081
        cx = 640
        fy = 827.856142111345
        cy = 400
        k1, k2, p1, p2, k3 = -0.335814019871572, 0.101431758719313, 0.0, 0.0, 0.0

        # 相机坐标系到像素坐标系的转换矩阵
        k = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        # 畸变系数
        d = np.array([
            k1, k2, p1, p2, k3
        ])
        h, w = frame.shape[:2]
        mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), 5)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

def save_image(img, rename_by_time, path="/img/"):
    '''
    True: 按照时间命名 False: 按照序号命名
    '''
    filename = sys.path[0] + path
    if rename_by_time:
        filename += time.strftime('%H%M%S')
    else:
        if 'index' not in globals():
            global index
            index = 1  # 保存图片起始索引
        else:
            index += 1
        filename += str(index)
    filename += ".jpg"
    # cv2.imwrite("./img/" + filename, img) # 非中文路径保存图片
    cv2.imencode('.jpg', img)[1].tofile(filename)  # 中文路径保存图片
    print("save img successfuly!")
    print(filename)