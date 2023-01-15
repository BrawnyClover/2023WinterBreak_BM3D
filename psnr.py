import numpy as np

def psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    RMSE = np.sqrt(np.sum((img1-img2)**2)/img1.size)
    return 20*np.log10(255./RMSE)