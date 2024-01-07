import time
import numpy as np


class CarController:
    def __init__(self):
        '''输入控制指令，暂存小车的速度、转角（方向加速度），按时计算位移
        '''
        self.speed = 0  # mm/s
        self.direction = 0  # deg, 0~360, 0:正北
        self.steer = 0
        self.timestamp = time.time()

    def read_command(self, T_para, command):
        '''
        T_para[1]: A/D
        T_para[2]: W/S
        '''
        self.speed += np.sign(T_para[2]) * 50  # mm/s
        self.speed = np.clip(self.speed, -500, 500)
        # self.steer += np.sign(T_para[1]) * 3  # deg/s/s 加速度
        self.steer = np.sign(T_para[1]) * 10  # deg/s
        self.steer = np.clip(self.steer, -40, 40)

        if command == 1:
            self.speed = 0
            self.direction = 0
            self.steer = 0

    def get_dT(self):
        dt = (time.time() - self.timestamp)
        self.timestamp = time.time()
        self.direction += self.steer * dt
        # self.direction = self.direction % 360
        dx = self.speed * dt * -np.cos(self.direction / 180 * np.pi)
        dy = self.speed * dt * np.sin(self.direction / 180 * np.pi)
        # print(f'speed: {self.speed}, steer: {self.steer}, direction: {self.direction}, dx: {dx}, dy: {dy}')
        return [[0, 0, self.steer * dt], dx, dy, 0]
