# Author: Xuechao Zhang
# Date: March 13th, 2022
# Description: 主入口

from src.virtual_cam import *
from src.xbox_controller import *
from src.monitor import Monitor

if __name__ == "__main__":
    env = VirtualCamEnv("config/cameras.yaml", 
                        "img/track.jpg", 
                        AprilTag_detection=False)
    monitor = Monitor({'topview': [0, 0, 800, 500],
                       'cam_1': [800, 0, 200, 200],
                       'cam_2': [800, 200, 200, 200],
                       })

    while True:
        # 缩放显示
        imgs = env.render() # 渲染图像
        monitor.show(imgs)
        # for (img, name) in imgs:
            # cv2.imshow(name, cv2.resize(img, (int(0.5*img.shape[1]), int(0.5*img.shape[0]))))

        env.control(*read_controller(CONTROLLER_TYPE=0)) # 读取控制器
