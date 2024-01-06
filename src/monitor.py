# Use one imshow window to display all images

from typing import List, Tuple, Dict
import numpy as np
import cv2

class Monitor:
    def __init__(self, layout: Dict[str, List[int]], size=[500, 1000]):
        '''
        layout: [[x, y, w, h], ...]
        '''
        self.size = size
        self.layout = layout
        for name in layout.keys():
            x, y, w, h = layout[name]
            if x + w > size[1] or y + h > size[0] or x < 0 or y < 0:
                raise ValueError(f'Invalid layout: {name} exceeds the size of the monitor')

    def show(self, imgs: List[Tuple[np.ndarray, str]]):
        panel = np.ones((self.size[0], self.size[1], 3), dtype=np.uint8) * 255
        for (img, name) in imgs:
            x, y, w, h = self.layout[name]
            source_h, source_w = img.shape[:2]
            ratio = min(w/source_w, h/source_h)
            img = cv2.resize(img, None, fx=ratio, fy=ratio)
            # 居中放置
            x = x + int((w - img.shape[1]) / 2)
            y = y + int((h - img.shape[0]) / 2)
            panel[y:y+img.shape[0], x:x+img.shape[1]] = img

        cv2.imshow('monitor', panel)
