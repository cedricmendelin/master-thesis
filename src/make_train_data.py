
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import imageio

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from odl import Operator
import odl
from tqdm import tqdm
import models
from utils.ODLHelper import OperatorFunction, OperatorModule


if torch.cuda.device_count()>1:
    torch.cuda.set_device(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_type=torch.float

data_CT_test = "data/limited-CT/data_png_test"
data_CT_train = "data/limited-CT/data_png_train"
resolution = 64
samples = 1024
SNR_min = 5
SNR_max = 5

batch_size = 32
lr = 1e-3
epochs = 500



#######################################
### Prepare dataset
#######################################
def find_SNR(ref, x):
    dif = torch.mean((ref-x)**2)
    nref = torch.mean(ref**2)
    return 10 * torch.log10((nref+1e-16)/(dif+1e-16))

def find_sigma_noise(SNR_value, x_ref):
    nref = torch.mean(x_ref**2)
    sigma_noise = (10**(-SNR_value/10)) * nref
    return torch.sqrt(sigma_noise)

# def add_noise_np(SNR, sinogram):
#     nref = np.mean(sinogram**2)
#     sigma_noise = (10**(-SNR/10)) * nref
#     sigma = np.sqrt(sigma_noise)
#     print("noise sigma:", sigma)
#     noise = np.random.randn(sinogram.shape[0], sinogram.shape[1]) * sigma
#     noisy_sino = sinogram + noise
#     return noisy_sino

# Train dataset
data_CT = data_CT_train
save_dir = "data/limited-CT/data_png_train_out"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

files = os.listdir(data_CT)

reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[resolution, resolution], dtype='float32')

# Angles: uniformly spaced, n = 1000, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, samples)

# Detector: uniformly sampled, n = 500, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, resolution)

# Make a parallel beam geometry with flat detector
geometry = odl.tomo.Parallel2dGeometry(angle_partition,detector_partition)

# Ray transform (= forward projection).
radon = odl.tomo.RayTransform(reco_space, geometry)

# Fourier transform in detector direction
fourier = odl.trafos.FourierTransform(radon.range, axes=[1])
# Create ramp in the detector direction
ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))
# Create ramp filter via the convolution formula with fourier transforms
ramp_filter = fourier.inverse * ramp_function * fourier

# Create filtered back-projection by composing the back-projection (adjoint)
# with the ramp filter.
fbp = radon.adjoint * ramp_filter

lin = np.linspace(-1,1,resolution)
XX, YY = np.meshgrid(lin,lin)
circle = ((XX**2+YY**2)<=1)*1.

for idx in range(1): # len(files)
    label = torch.from_numpy((imageio.imread(data_CT+os.sep+files[idx])/255)*circle)
    data = OperatorFunction.apply(radon, label).data

    # add noise
    SNR = np.random.rand()*(SNR_max-SNR_min)+SNR_min
    sigma = find_sigma_noise(SNR, data)
    data_ = data.clone()
    data = data + torch.randn_like(data)*sigma
    print("SNR:",find_SNR(data_, data).detach().cpu().numpy())

    # reconstruct
    data = OperatorFunction.apply(fbp,data.view(1, samples, resolution))
    data = data.view(resolution,resolution).type(torch_type)

    out = data.detach().cpu().numpy()
    out = (out-out.min())/(out.max()-out.min())
    imageio.imwrite(save_dir+os.sep+files[idx],np.clip(out*255,0,255).astype(np.uint8))


# Test dataset
data_CT = data_CT_test
save_dir = "data/limited-CT/data_png_test_out"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
## TODO: same for test