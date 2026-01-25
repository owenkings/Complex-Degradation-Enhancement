import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def batch_psnr_ssim(clean_batch, enhanced_batch):
    """
    clean_batch, enhanced_batch: torch.Tensor, shape (B, C, H, W), [0,1] 区间
    返回: (mean_psnr, mean_ssim)
    """
    clean_np = clean_batch.detach().cpu().numpy()
    enh_np = enhanced_batch.detach().cpu().numpy()

    psnrs, ssims = [], []
    B = clean_np.shape[0]
    for i in range(B):
        c = np.transpose(clean_np[i], (1, 2, 0))  # (H, W, C)
        e = np.transpose(enh_np[i], (1, 2, 0))
        psnr = peak_signal_noise_ratio(c, e, data_range=1.0)
        ssim = structural_similarity(c, e, data_range=1.0, channel_axis=-1)
        psnrs.append(psnr)
        ssims.append(ssim)

    return float(np.mean(psnrs)), float(np.mean(ssims))
