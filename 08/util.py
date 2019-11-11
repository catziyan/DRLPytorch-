import cv2
import numpy as np


def preprocess(observation):
    img = np.reshape(observation, [210, 160, 3]).astype(np.float32)
    # RGB转换成灰度图像的一个常用公式是：ray = R*0.299 + G*0.587 + B*0.114
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114  # shape (210,160)
    resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)  # shape(110,84)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    x_t.astype((np.uint8))
    x_t = np.moveaxis(x_t, 2, 0)  # shape（1，84，84）
    return np.array(x_t).astype(np.float32) / 255.0

