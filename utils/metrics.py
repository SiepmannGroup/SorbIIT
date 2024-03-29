import numpy as np
import torch
import torch.nn.functional as F

# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]).float()
    return gauss/gauss.sum()

def create_window(window_size, channel):
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = (window_1d @ window_1d.t()).float()
    window_3d = (window_2d.unsqueeze(2) @ window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_3d.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map


def ssim(img1, img2, window_size = 11):
    channel = img1.shape[1]
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.device)
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel)
