# Author: Xuechao Zhang
# Date: March 13th, 2022
# Description: 主入口

from src.virtual_cam import *
from src.xbox_controller import *
from src.monitor import Monitor
from src.line_tracker import *
from multiprocessing import Pool


if __name__ == "__main__":
    env = VirtualCamEnv("config/cameras.yaml", 
                        "img/track.jpg", 
                        AprilTag_detection=False)
    monitor = Monitor({'topview': [0, 0, 800, 450],
                       'cam_1': [800, 0, 200, 150],
                    #    'cam_2': [800, 150, 200, 150],
                       'mask': [800, 150, 200, 150],
                       'annotated': [800, 300, 200, 150],
                       }, size=[450, 1000])

    # car_controller = CarController()
    # car_controller.read_command(T_para)

    linetracker = LineTrackerCV(delay=False)

    while True:
        # 缩放显示
        imgs = env.render() # 渲染图像

        steer, log_imgs = linetracker.process(imgs['cam_1'])
        imgs.update(log_imgs)
        monitor.show(imgs)

        T_para, command = read_controller(CONTROLLER_TYPE=0)
        T_para[1] = steer
        env.control(T_para, command) # 读取控制器
