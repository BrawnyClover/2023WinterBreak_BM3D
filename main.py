import cv2
import numpy as np
from consts import *
from bm3d_step_1 import bm3d_step_1
from bm3d_step_2 import bm3d_step_2
from psnr import psnr

images = ["cameraman","house",  "lena","barbara"]
noises = [10, 25, 35, 75, 100]
f = None

def get_target_img(image_title):
    img = cv2.imread(image_title + ".png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def get_noisy_img(img):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    return noisy

def denoise(image_title, img):
    noisy_img = get_noisy_img(img)
    cv2.imwrite(f'{image_title}_{sigma}_noisy.png', noisy_img)
    
    step_1_img = bm3d_step_1(noisy_img)
    cv2.imwrite(f'{image_title}_{sigma}_step1.png', step_1_img)
    print("step 1 done")
    # psnr_basic = psnr(img, step_1_img)
    # step_1_img = cv2.imread('step1.png')
    # step_1_img = cv2.cvtColor(step_1_img, cv2.COLOR_BGR2GRAY)

    step_2_img = bm3d_step_2(noisy_img, step_1_img)
    psnr_wiener = psnr(img, step_2_img)
    result_img = step_2_img
    cv2.imwrite(f'{image_title}_{sigma}_result.png', result_img)

    # print(f'PSNR basic : {psnr_basic}')
    f.write(f'{image_title}, {sigma}, {psnr_wiener}\n')

if __name__ == "__main__":
    f = open("result.txt", 'w')
    for image_title in images:
        for noise in noises:
            sigma = noise
            if sigma <= 40:
                lambda_2d = 0
                N_wien = 8
                N_hard = 8
            elif sigma > 40:
                lambda_2d = 2
                N_hard = 12
                N_wien = 11
            print(f'{image_title} {sigma}')
            img = get_target_img(image_title)
            denoise(image_title, img)
        f.write('\n')

    f.close()


