import cv2
import numpy as np

# 一个基类，输入图像，输出转向角
class LineTracker:
    def __init__(self):
        pass
    
    def process(self, img):
        '''输入图像，输出转向角（要求子类实现）
        '''
        raise NotImplementedError

# 基于传统 CV 的方法
class LineTrackerCV(LineTracker):
    def __init__(self):
        super().__init__()
    
    def process(self, frame):
        binary, gray = self._get_binary_img(frame)
        mask, area = self._find_largest_component(binary)
        log_imgs = {'gray': gray,
                    'binary': binary,
                    'mask': mask}
        if area > 0:
            annotated, a, b, residuals = self._fit_line(frame.copy(), binary)
            print(a, b, residuals)
            log_imgs['annotated'] = annotated
            steer = 0.5 * a + 0.5 * b
            return steer, log_imgs
        else:
            print("Found no white lane!")
            return 0, log_imgs

    def _get_binary_img(self, origin, threshold=130):
        # 灰度
        # gray_ave = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
        gray_min = np.min(origin, axis=2)  # 把 origin 转成一通道的，灰度值选择 RGB 三通道中的最小值
        gray_max = np.max(origin, axis=2)
        gray_min = gray_min - 0.5 * (gray_max - gray_min)
        gray_min = np.uint8(gray_min)
        # gray_min[gray_min < 0] = 255

        # 调整曝光
        # equalized = cv2.equalizeHist(gray)  # 直方图均衡化，不好用
        image = gray_min.astype(np.float32) / 255.0

        mean_value = np.mean(image)
        # print("mean_value: ", mean_value)
        brightness_factor = 0.37 / mean_value
        adjusted_image = cv2.multiply(image, brightness_factor)
        # print("after_mean_value: ", np.mean(adjusted_image))
        adjusted_image = adjusted_image * 255
        adjusted_image[adjusted_image > 255] = 255
        adjusted_image[adjusted_image < 0] = 0
        adjusted_image = adjusted_image.astype(np.uint8)
        
        # 二值化
        # binary_adp = cv2.adaptiveThreshold(adjusted_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
        # t, binary_adap = cv2.threshold(adjusted_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # print("threshold: ", t)  # 直方图双峰类型的图片挺合适的
        # [4.3. 图像阈值 - OpenCV Python Tutorials]( https://opencv-python-tutorials.readthedocs.io/zh/latest/4.%20OpenCV%E4%B8%AD%E7%9A%84%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/4.3.%20%E5%9B%BE%E5%83%8F%E9%98%88%E5%80%BC/ )
        _, binary = cv2.threshold(adjusted_image, threshold, 255, cv2.THRESH_BINARY)
        return binary, adjusted_image

    def _find_largest_component(self, binary, min_area=400):
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        if len(stats) > 1:
            largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # 找到最大连通域的索引
            mask = np.uint8(labels == largest_component_index) * 255  # 创建一个只包含最大连通域的掩码
            area = np.sum(mask) / 255  # 计算 mask 的面积
            if area > min_area:
                # 绘制最大连通域的边界框
                # x, y, w, h = stats[largest_component_index, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4]
                # cv2.rectangle(binary, (x, y), (x + w, y + h), (255, 255, 255), 2)
                return mask, area

        return np.zeros_like(binary), 0

    def _fit_line(self, origin, binary, circle_size=10):
        # 记录中点
        mid_points = []

        # 遍历每一行像素的第一个白点和最后一个白点
        for row in range(binary.shape[0]):
            left = -1
            right = -1
            for col in range(binary.shape[1]):
                if binary[row, col] == 255:
                    left = col
                    break
            for col in range(binary.shape[1] - 1, 0, -1):
                if binary[row, col] == 255:
                    right = col
                    break
            # 在原图上绘制出这两个点
            # origin[row, left] = (0, 0, 255)
            cv2.circle(origin, tuple((left, row)), circle_size, (0, 0, 255), 2)
            # origin[row, right] = (255, 0, 0)
            cv2.circle(origin, tuple((right, row)), circle_size, (255, 0, 0), 2)

            if left != -1 and right != -1:
                # 计算出中心点
                center = (left + right) // 2
                mid_points.append((row, center))
                # 在原图上绘制出这个中心点
                # origin[row, center] = (0, 255, 0)
                cv2.circle(origin, tuple((center, row)), circle_size, (0, 255, 0), 2)

        # 最小二乘拟合中点
        x = np.array([point[0] for point in mid_points])  # 生成 x 和 y 的数组
        y = np.array([point[1] for point in mid_points])
        z1 = np.polyfit(x, y, 1)  # 最小二乘拟合
        p1 = np.poly1d(z1)  # 生成拟合函数 (分别是斜率和截距)

        # 计算误差
        residuals = y - p1(x)

        x_new = np.linspace(x.min(), x.max(), 100)  # 生成拟合后的 x 和 y 的数组
        y_new = p1(x_new)
        for i in range(len(x_new)):  # 标注拟合后的中心线
            if 1 < int(y_new[i]) < binary.shape[1]-1:
                cv2.circle(origin, tuple((int(y_new[i]), int(x_new[i]))), circle_size, (50, 50, 100), 2)
                # origin[int(x_new[i]), int(y_new[i])-1] = (50, 50, 100)
                # origin[int(x_new[i]), int(y_new[i])] = (50, 50, 100)
                # origin[int(x_new[i]), int(y_new[i])+1] = (50, 50, 100)
        # 计算转向（左负右正）
        lane_angle = -np.arctan(z1[0]) * 180 / np.pi
        horizontal_offset = (y_new[-1]/binary.shape[1] - 0.5) * 20
        # 计算角度
        # steering = angle / 90
        # 标注角度
        # cv2.putText(origin, "angle: {:.2f}".format(angle), (10, 10),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 100, 255), 1)
        return origin, lane_angle, horizontal_offset, np.mean(abs(residuals))
