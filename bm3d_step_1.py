from consts import *
import numpy as np
from scipy.fftpack import dct, idct

def get_window_coord(img_size, ref_point):
    ret_val = np.zeros((2,2), dtype = int)
    ret_val[0, 0] = max(0, ref_point[0]+int((N_hard-window_size)/2)) # left-top x
    ret_val[0, 1] = max(0, ref_point[1]+int((N_hard-window_size)/2)) # left-top y               
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
    overall_dct = np.zeros((img.shape[0]-N_hard, img.shape[1]-N_hard, N_hard, N_hard),dtype = float)
    for i in range(overall_dct.shape[0]):
        for j in range(overall_dct.shape[1]):
            target_block = img[i:i+N_hard, j:j+N_hard]
            overall_dct[i, j, :, :] = dct2D(target_block.astype(np.float64))
            
    return overall_dct

def grouping(img, x_R, overall_dct):
    ref_block = overall_dct[x_R[0], x_R[1], :, :]
    # dct_ref_block = dct2D(ref_block)
    height = (window_size - N_hard + 1) ** 2
    block_group = np.zeros((height, N_hard, N_hard), dtype = float)
    block_pos = np.zeros((height, 2), dtype = int)
    window_coord = get_window_coord(img.shape, x_R)

    dist = np.zeros(height, dtype=float)
    cnt = 0

    for i in range(window_size - N_hard + 1):
        for j in range(window_size - N_hard + 1):
            target_block = overall_dct[window_coord[0, 0]+i, window_coord[0, 1]+j, :, :]
            
            # dct_target_block = dct2D(target_block)
            d = calc_dist(ref_block , target_block)
            if d < tau_hard:
                # print(target_block)
                block_pos[cnt, :] = [window_coord[0, 0]+i, window_coord[0, 1]+j]
                block_group[cnt, :, :] = target_block
                dist[cnt] = d
                cnt = cnt + 1
    # print(dist)
    if cnt > max_patch:
        dist_sorted_idx = np.argpartition(dist[:cnt], max_patch)
        # print(dist_sorted_idx)
        block_pos = block_pos[dist_sorted_idx[:max_patch], :]
        block_group = block_group[dist_sorted_idx[:max_patch], :]
    
    else:
        block_pos = block_pos[:cnt, :]
        block_group = block_group[:cnt, :, :]
    # print(block_group)
    return block_pos, block_group

def calc_dist(ref_dct, target_dct):
    if sigma > 40:
        threshold_value = lambda_2d * sigma
        ref_dct = np.where(abs(ref_dct) < threshold_value, 0, ref_dct)
        target_dct = np.where(abs(target_dct) < threshold_value, 0, target_dct)

    return np.linalg.norm(ref_dct - target_dct)**2 / (N_hard**2)


def dct2D(block):
    return dct(dct(block, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')

def idct2D(dct_block):
    return idct(idct(dct_block, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')

def thresholding(target, threshold):
    if abs(target) < threshold :
        return 0
    return target

def filtering3d(block_group):
    threshold = lambda_3d * sigma
    # print(threshold)
    non_zero_cnt = 0
    for i in range(block_group.shape[1]):
        for j in range(block_group.shape[2]):
            dct_res = dct(block_group[:,i,j], norm="ortho")
            # print(dct_res)
            dct_res[abs(dct_res[:] < threshold)] = 0.
            non_zero_cnt += np.nonzero(dct_res)[0].size
            block_group[:,i,j] = list((idct(dct_res, norm="ortho")))
    # print(non_zero_cnt)
    return block_group, non_zero_cnt

def aggregation(target_img, target_weight_basic, block_group, block_pos, non_zero_cnt, basic_kaiser):
    if non_zero_cnt >= 1:
        aggr_weight = (float(1.0)/(sigma**2 * non_zero_cnt)) * basic_kaiser
    else:
        aggr_weight = 1 * basic_kaiser

    for i in range(block_pos.shape[0]):
        target_img[block_pos[i, 0] : block_pos[i, 0] + block_group.shape[1], block_pos[i, 1] : block_pos[i, 1] + block_group.shape[2]] += aggr_weight * idct2D(block_group[i, :, :])
        target_weight_basic[block_pos[i, 0]:block_pos[i, 0]+block_group.shape[1], block_pos[i, 1]:block_pos[i, 1]+block_group.shape[2]] += aggr_weight

def bm3d_step_1(img):

    kaiser_window = np.matrix(np.kaiser(N_hard, 2.0))
    basic_kaiser = np.array(kaiser_window.T * kaiser_window)

    step_1_result_img = np.zeros(img.shape, dtype=float)
    step_1_weight_basic = np.zeros(img.shape, dtype=float)
    overall_dct = get_overall_dct(img)
    size = int((img.shape[0] - N_hard)/speed_up)+2
    for i in range(size):
        print(f'step 1 : {i}/{size}')
        for j in range(size):
            x = min(speed_up * i, img.shape[0] - N_hard - 1)
            y = min(speed_up * j, img.shape[1] - N_hard - 1)
            x_R = [x, y]
            # print(x_R)
            block_pos, block_group = grouping(img, x_R, overall_dct)
            block_group, non_zero_cnt = filtering3d(block_group)

            aggregation(step_1_result_img, step_1_weight_basic, block_group, block_pos, non_zero_cnt, basic_kaiser)
    
    step_1_weight_basic = np.where(step_1_weight_basic == 0, 1, step_1_weight_basic)
    
    step_1_result_img[:, :] /= step_1_weight_basic[:, :]
    return step_1_result_img