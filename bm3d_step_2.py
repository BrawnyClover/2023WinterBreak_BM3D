from consts import *
import numpy as np
from scipy.fftpack import dct, idct

def get_overall_dct(img):
    overall_dct = np.zeros((img.shape[0]-N_hard, img.shape[1]-N_hard, N_hard, N_hard),dtype = float)
    for i in range(overall_dct.shape[0]):
        for j in range(overall_dct.shape[1]):
            target_block = img[i:i+N_hard, j:j+N_hard]
            overall_dct[i, j, :, :] = dct2D(target_block.astype(np.float64))
            
    return overall_dct

def grouping(x_R, basic, basic_dct, noisy, noisy_dct):
    height = (basic.shape[0] - N_wien + 1) ** 2
    
    block_pos = np.zeros((height, 2), dtype = int)
    block_group_basic = np.zeros((height, N_wien, N_wien), dtype = int)
    block_group_noisy = np.zeros((height, N_wien, N_wien), dtype = int)

    dist = np.zeros(height, dtype=float)
    cnt = 0

    ref_block = basic[x_R[0]:x_R[0]+N_wien, x_R[1]:x_R[1]+N_wien]

    for i in range(basic.shape[0] - N_wien):
        for j in range(basic.shape[1] - N_wien):
            target_pos = [i, j]
            target_block = basic[target_pos[0]:target_pos[0]+N_wien, target_pos[1]:target_pos[1]+N_wien]
            d = calc_dist(ref_block, target_block)
            if d < tau_wien:
                # print(target_block)
                block_pos[cnt, :] = target_pos
                dist[cnt] = d
                cnt = cnt + 1
    # print(dist)
    if cnt > max_patch:
        dist_sorted_idx = np.argsort(dist[:cnt])
        block_pos = block_pos[dist_sorted_idx[:max_patch], :]
    
    else:
        block_pos = block_pos[:max_patch, :]
    
    for i in range(block_pos.shape[0]):
        target_pos = block_pos[i, :]
        block_group_basic[i,:,:] = basic_dct[target_pos[0], target_pos[1], :,:]
        block_group_noisy[i,:,:] = noisy_dct[target_pos[0], target_pos[1], :,:]
    
    block_group_basic = block_group_basic[:block_pos.shape[0],:,:]
    block_group_noisy = block_group_noisy[:block_pos.shape[0],:,:]
    
    return block_pos, block_group_basic, block_group_noisy

def calc_dist(ref_block, target_block):
    return np.linalg.norm(ref_block - target_block)**2 / (N_wien**2)


def dct2D(block):
    return dct(dct(block, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')

def idct2D(dct_block):
    return idct(idct(dct_block, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')

def filtering3d(block_group_basic, block_group_noisy):
    
    # 1. block_group_basic으로 wiener coefficient 구하기
    # 2. block_group_noisy를 3D DCT한 것에 wiener coeff를 곱하고 idct하기
    weight = 0
    for i in range(block_group_noisy.shape[1]):
        for j in range(block_group_noisy.shape[2]):
            dct_3d_basic = dct(block_group_basic[:, i, j], norm='ortho')
            dct_3d_noisy = dct(block_group_noisy[:, i, j], norm='ortho')
            wiener_coeff = dct_3d_basic**2 / (dct_3d_basic**2 + sigma**2)
            dct_3d_noisy *= wiener_coeff
            block_group_noisy[:, i, j] = list(idct(dct_3d_noisy, norm='ortho'))
            weight += np.sum(wiener_coeff)
    # print(non_zero_cnt)

    if weight > 0:
        wiener_weight = 1.0 / ((sigma**2) * pow(np.linalg.norm(weight), -2))
    else :
        wiener_weight = 1.0
    return block_group_noisy, wiener_weight

def aggregation(target_img, target_weight_basic, block_group, block_pos, wiener_weight):
    
    for i in range(block_pos.shape[0]):
        target_img[block_pos[i, 0] : block_pos[i, 0] + block_group.shape[1], block_pos[i, 1] : block_pos[i, 1] + block_group.shape[2]] += wiener_weight * idct2D(block_group[i, :, :])
        target_weight_basic[block_pos[i, 0]:block_pos[i, 0]+block_group.shape[1], block_pos[i, 1]:block_pos[i, 1]+block_group.shape[2]] += wiener_weight

def bm3d_step_2(noisy, basic):
    step_2_result_img = np.zeros(noisy.shape, dtype=float)
    step_2_weight_basic = np.zeros(noisy.shape, dtype=float)
    basic_dct = get_overall_dct(basic)
    noisy_dct = get_overall_dct(noisy)
    for i in range(int((noisy.shape[0] - N_hard)/speed_up)):
        for j in range(int((noisy.shape[1] - N_hard)/speed_up)):
            x = min(speed_up * i, noisy.shape[0] - N_hard - 1)
            y = min(speed_up * j, noisy.shape[1] - N_hard - 1)
            x_R = [x, y]
            print(x_R)
            block_pos, block_group_basic, block_group_noisy = grouping(x_R, basic, basic_dct, noisy, noisy_dct)
            block_group_noisy, wiener_weight = filtering3d(block_group_basic, block_group_noisy)

            aggregation(step_2_result_img, step_2_weight_basic, block_group_noisy, block_pos, wiener_weight)
    
    step_2_weight_basic = np.where(step_2_weight_basic == 0, 1, step_2_weight_basic)
    
    step_2_result_img[:, :] /= step_2_weight_basic[:, :]
    return step_2_result_img