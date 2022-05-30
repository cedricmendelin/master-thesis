import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np

from utils.Plotting import *
from utils.ODLHelperCustom import setup_forward_and_backward
from utils.SNRHelper import *
from utils.Graph import *

from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
from scipy.spatial import distance_matrix
from sklearn.neighbors import kneighbors_graph

from utils.UNetModel import UNet

torch.manual_seed(2022)
np.random.seed(2022)



resolution = 64
samples = 1024
snr = 10
phantom = shepp_logan_phantom()
phantom = rescale(phantom, scale=resolution / phantom.shape[0], mode='reflect')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load("models/unet.pt", map_location=device)
unet = UNet(nfilter=128).eval()
unet.load_state_dict(checkpoint['model_state_dict'])    

radon, fbp =  setup_forward_and_backward(resolution, samples)

sino = radon(phantom).data
sino_noisy = add_noise_np(snr, sino)

reconstruction = fbp(sino)
reconstruction_snr = fbp(sino_noisy)



import bm3d

std = np.std(sino - sino_noisy)
denoised_sino_std = bm3d.bm3d(sino_noisy, sigma_psd=std, stage_arg=bm3d.BM3DStages.ALL_STAGES)

print(np.linalg.norm(denoised_sino_std - sino))
print(np.linalg.norm(sino_noisy - sino))

std2 = np.std(phantom - reconstruction_snr)

print(std2)
denoised_reco_std = bm3d.bm3d(reconstruction_snr, sigma_psd=std2, stage_arg=bm3d.BM3DStages.ALL_STAGES)

print(np.linalg.norm(denoised_reco_std - phantom))
print(np.linalg.norm(reconstruction_snr - phantom))
print(np.linalg.norm(reconstruction - phantom))

plot_imshow(phantom, title="Shepp Logan Phantom", colorbar=False)
plot_imshow(sino, title="Sinogram", xlabel="s", ylabel='$\\theta$', colorbar=False)
plot_imshow(sino_noisy, title=f"Sinogram with noise SNR: {snr} dB", xlabel="s", ylabel='$\\theta$', colorbar=False)
plot_imshow(reconstruction, title="FBP clean sinogram", colorbar=False)
plot_imshow(reconstruction_snr, title="FBP noisy sinogram", colorbar=False)
plot_imshow(fbp(denoised_sino_std).data, title="FBP BM3d denoised std", colorbar=False)
plot_imshow(denoised_reco_std, title="BM3d denoised reco std", colorbar=False)


# plt.show()