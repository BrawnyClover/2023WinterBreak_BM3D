import cv2
import numpy as np
from consts import *
from bm3d_step_1 import bm3d_step_1
from bm3d_step_2 import bm3d_step_2

def get_target_img():
    img = cv2.imread('lena.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def get_noisy_img(img):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    return noisy

if __name__ == "__main__":
    img = get_target_img()
    noisy_img = get_noisy_img(img)
    cv2.imwrite("noisy.png", noisy_img)
    step_1_img = bm3d_step_1(noisy_img)
    cv2.imwrite("step1.png", step_1_img)

    # step_1_img = cv2.imread('step1.png')
    # step_1_img = cv2.cvtColor(step_1_img, cv2.COLOR_BGR2GRAY)

    step_2_img = bm3d_step_2(noisy_img, step_1_img)
    result_img = step_2_img
    cv2.imwrite("result.png", result_img)


