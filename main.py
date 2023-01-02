import cv2
from consts import *
from bm3d_step_1 import bm3d_step_1

def get_target_img():
    img = cv2.imread('noise.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

if __name__ == "__main__":
    noisy_img = get_target_img()
    step_1_img = bm3d_step_1(noisy_img)
    # step_2_img = bm3d_step_2(step_1_img)
    result_img = step_1_img
    cv2.imwrite("result.png", result_img)


