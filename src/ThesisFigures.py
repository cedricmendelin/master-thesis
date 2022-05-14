import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np

from utils.Plotting import *
from utils.ODLHelperCustom import setup_forward_and_backward
from utils.SNRHelper import *

from skimage.data import shepp_logan_phantom
from skimage.transform import rescale


def chapter_imaging_sinos():
  resolution = 400
  samples = 500
  snr = 10
  phantom = shepp_logan_phantom()
  phantom = rescale(phantom, scale=resolution / phantom.shape[0], mode='reflect')



  radon, fbp =  setup_forward_and_backward(resolution, samples)

  sino = radon(phantom)
  sino_noisy = add_noise_np(snr, sino)
  reconstruction = fbp(sino)
  reconstruction_snr = fbp(sino_noisy)


  plot_imshow(phantom, title="Shepp Logan Phantom", colorbar=False)
  plot_imshow(sino, title="Sinogram", xlabel="s", ylabel='$\\theta$', colorbar=False)
  plot_imshow(sino_noisy, title=f"Sinogram with noise SNR: {snr}", xlabel="s", ylabel='$\\theta$', colorbar=False)
  plot_imshow(reconstruction, title="FBP clean sinogram", colorbar=False)
  plot_imshow(reconstruction_snr, title="FBP noisy sinogram", colorbar=False)

  plt.show()


chapter_imaging_sinos()