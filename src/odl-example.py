"""Example for reconstruction with FBP in 2d parallel geometry.

This example creates a filtered back-projection operator in 2d using the
ray transform and a ramp filter. This ramp filter is implemented in Fourier
space.

See https://en.wikipedia.org/wiki/Radon_transform#Inversion_formulas for
more information.

Also note that ODL has a utility function, `fbp_op` that can be used to
generate the FBP operator. This example is intended to show how the same
functionality could be implemented by hand in ODL.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import odl
import torch

import matplotlib as plt
from utils.Plotting import *


# --- Set up geometry of the problem --- #

def _add_noise(SNR, sinogram):
    VARIANCE = 10 ** (-SNR/10) * (torch.std(sinogram) ** 2)
    noise = torch.randn(
        sinogram.shape[0], sinogram.shape[1]) * torch.sqrt(VARIANCE)
    return sinogram + noise

# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[128, 128], dtype='float32')

# Angles: uniformly spaced, n = 1000, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 1024)

# Detector: uniformly sampled, n = 500, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 128)

# Make a parallel beam geometry with flat detector
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)


# --- Create Filtered Back-projection (FBP) operator --- #


# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# Fourier transform in detector direction
fourier = odl.trafos.FourierTransform(ray_trafo.range, axes=[1])

# Create ramp in the detector direction
ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))

# Create ramp filter via the convolution formula with fourier transforms
ramp_filter = fourier.inverse * ramp_function * fourier

# Create filtered back-projection by composing the back-projection (adjoint)
# with the ramp filter.
fbp = ray_trafo.adjoint * ramp_filter


# --- Show some examples --- #
import imageio
val_image = imageio.imread("src/val_image.png")
if val_image.ndim > 2:  # passes to grayscale if color
    val_image = val_image[:, :, 0]

val_image = val_image[0:128, 0:128]/255
noisy_val = _add_noise(-5, torch.from_numpy(val_image))

sino_val = ray_trafo(val_image).data
sino_noisy_val = ray_trafo(noisy_val).data
sino_noisy = _add_noise(-5, torch.from_numpy(sino_val))

# # Create a discrete Shepp-Logan phantom (modified version)
# #phantom = odl.phantom.shepp_logan(reco_space, modified=True)
# noisy_phantom = _add_noise(10, torch.from_numpy(phantom.data))

# sino = ray_trafo(phantom).data
# sino_noisy_phantom = ray_trafo(noisy_phantom).data
# sino_noisy = _add_noise(10, torch.from_numpy(sino))

import bm3d
denoised_sino = bm3d.bm3d(sino_noisy, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)
denoised_val = bm3d.bm3d(noisy_val, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.ALL_STAGES)

plot_imshow(val_image)
plot_imshow(denoised_val, "BM3D on noisy input image")
# plot_imshow(sino, "Clean sino")
# plot_imshow(sino_noisy_phantom, "Sino from noisy phantom")
# plot_imshow(sino_noisy, "Noisy sino")
# plot_imshow(denoised_sino , "BM3D denoised sino")

# print(torch.linalg.norm(torch.from_numpy(sino) - sino_noisy_phantom), "sino noisy phantom loss")
# print(torch.linalg.norm(torch.from_numpy(sino) - sino_noisy), "sino noisy loss")
# print(torch.linalg.norm(torch.from_numpy(sino) - torch.from_numpy(denoised_sino)), "sino denoised loss")

# phantom_rec = fbp(sino)
phantom_rec_noisy_phantom = fbp(sino_noisy_val)
phantom_rec_noisy = fbp(sino_noisy)
phantom_denoised_bm3d = fbp(denoised_sino)


phantom_rec_noisy_phantom.show(title="noisy phantom FBP")
phantom_rec_noisy.show(title="noisy sino FBP")
plot_imshow(phantom_denoised_bm3d, title="Denoised bm3d FBP")


print(np.linalg.norm(val_image - phantom_rec_noisy_phantom))
print(np.linalg.norm(val_image - phantom_rec_noisy))


plt.show()

# # phantoms = torch.cat((phantom, phantom)).view(2,1,300,300)
# # self.T_input_images.view(self.M, 1, self.RESOLUTION, self.RESOLUTION))

# print("phantoms done")
# # Create projection data by calling the ray transform on the phantom
# proj_data = ray_trafo(phantom)

# print("projections done")

# print(proj_data)

# t_proj_data = torch.from_numpy(proj_data.data)

# print(t_proj_data.size())

# print("tensor done")

# # sinos = torch.cat([torch.tensor(ray_trafo(phantom)) for i in range(5)])

# sinos = torch.cat([t_proj_data, t_proj_data])

# print("cat sinos")

# sinos = sinos.view(2,1, 1024, 192)

# # Calculate filtered back-projection of data
# fbp_reconstruction = fbp(sinos[0,0])

# # Shows a slice of the phantom, projections, and reconstruction
# phantom.show(title='Phantom')
# proj_data.show(title='Projection Data (Sinogram)')
# fbp_reconstruction.show(title='Filtered Back-projection')
# (phantom - fbp_reconstruction).show(title='Error', force_show=True)