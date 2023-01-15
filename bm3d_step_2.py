from consts import *
import numpy as np
from scipy.fftpack import dct, idct

def get_window_coord(img_size, ref_point):
    ret_val = np.zeros((2,2), dtype = int)
    ret_val[0, 0] = max(0, ref_point[0]+int((N_wien-window_size)/2)) # left-top x
    ret_val[0, 1] = max(0, ref_point[1]+int((N_wien-window_size)/2)) # left-top y               
    ret_val[1, 0] = ret_val[0, 0] + window_size # right-bottom x
    ret_val[1, 1] = ret_val[0, 1] + window_size # right-bottom y             
    if ret_val[1, 0] >= img_size[0]:
        ret_val[1, 0] = img_size[0] - 1
        ret_val[0, 0] = ret_val[1, 0] - window_size
    if ret_val[1, 1] >= img_size[1]:
        ret_val[1, 1] = img_size[1] - 1
        ret_val[0, 1] = ret_val[1, 1] - window_size
    return ret_val

def get_overall_dct(img):
    overall_dct = np.zeros((img.shape[0]-N_wien, img.shape[1]-N_wien, N_wien, N_wien),dtype = float)
    for i in range(overall_dct.shape[0]):
        for j in range(overall_dct.shape[1]):
            target_block = img[i:i+N_wien, j:j+N_wien]
            overall_dct[i, j, :, :] = dct2D(target_block.astype(np.float64))
            
    return overall_dct

def grouping(x_R, basic, basic_dct, noisy_dct):
    height = (window_size - N_wien + 1) ** 2
    
    block_pos = np.zeros((height, 2), dtype = int)
    block_group_basic = np.zeros((height, N_wien, N_wien), dtype = int)
    block_group_noisy = np.zeros((height, N_wien, N_wien), dtype = int)

    dist = np.zeros(height, dtype=float)
    cnt = 0
    window_coord = get_window_coord(basic.shape, x_R)
    ref_block = basic[x_R[0]:x_R[0]+N_wien, x_R[1]:x_R[1]+N_wien].astype(np.float64)

    size = window_size - N_wien + 1
    for i in range(size):
        for j in range(size):
            target_pos = [window_coord[0, 0]+i, window_coord[0, 1]+j]
            target_block = basic[target_pos[0]:target_pos[0]+N_wien, target_pos[1]:target_pos[1]+N_wien].astype(np.float64)
            # print(f'{target_pos[0]}:{target_pos[0]+N_wien}, {target_pos[1]}:{target_pos[1]+N_wien}')
            # print(window_coord[0], end=', ')
            # print(i, end=', ')
            # print(j)
            d = calc_dist(ref_block, target_block)
            if d < tau_wien:
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

def aggregation(target_img, target_weight_basic, block_group, block_pos, wiener_weight, final_kaiser):
    wiener_weight = wiener_weight * final_kaiser
    for i in range(block_pos.shape[0]):
        target_img[block_pos[i, 0] : block_pos[i, 0] + block_group.shape[1], block_pos[i, 1] : block_pos[i, 1] + block_group.shape[2]] += wiener_weight * idct2D(block_group[i, :, :])
        target_weight_basic[block_pos[i, 0]:block_pos[i, 0]+block_group.shape[1], block_pos[i, 1]:block_pos[i, 1]+block_group.shape[2]] += wiener_weight

def bm3d_step_2(noisy, basic):
    kaiser_window = np.matrix(np.kaiser(N_wien, 2.0))
    final_kaiser = np.array(kaiser_window.T * kaiser_window)

    step_2_result_img = np.zeros(noisy.shape, dtype=float)
    step_2_weight_basic = np.zeros(noisy.shape, dtype=float)
    basic_dct = get_overall_dct(basic)
    noisy_dct = get_overall_dct(noisy)
    size = int((noisy.shape[0] - N_wien)/speed_up)+2
    for i in range(size):
        print(f'step 2 : {i}/{size}')
        for j in range(size):
            x = min(speed_up * i, noisy.shape[0] - N_wien - 1)
            y = min(speed_up * j, noisy.shape[1] - N_wien - 1)
            x_R = [x, y]
            # print(x_R)
            block_pos, block_group_basic, block_group_noisy = grouping(x_R, basic, basic_dct, noisy_dct)
            block_group_noisy, wiener_weight = filtering3d(block_group_basic, block_group_noisy)

            aggregation(step_2_result_img, step_2_weight_basic, block_group_noisy, block_pos, wiener_weight, final_kaiser)
    
    step_2_weight_basic = np.where(step_2_weight_basic == 0, 1, step_2_weight_basic)
    
    step_2_result_img[:, :] /= step_2_weight_basic[:, :]
    return step_2_result_img