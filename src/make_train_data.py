
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
import utils.UNetModel as UNetModel
from utils.ODLHelper import OperatorFunction, OperatorModule

if torch.cuda.device_count()>1:
    torch.cuda.set_device(1)


"""
This script generate noisy reconstruction to train the Unet to remot arctefact and denoise.
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_type=torch.float

data_CT_test = "data/limited-CT/data_png_test"
data_CT_train = "data/limited-CT/data_png_train"
resolution = 64
samples = 1024
SNR_min = -10
SNR_max = 0

batch_size = 32
lr = 1e-3
epochs = 500



#######################################
### SNR functions
#######################################
def find_SNR(ref, x):
    dif = torch.std((ref-x))**2
    nref = torch.std(ref)**2
    return 10 * torch.log10((nref+1e-16)/(dif+1e-16))

def find_sigma_noise(SNR_value, x_ref):
    nref = torch.std(x_ref)**2
    sigma_noise = (10**(-SNR_value/10)) * nref
    return torch.sqrt(sigma_noise)


#######################################
### Prepare Forward
#######################################
reco_space = odl.uniform_discr(
    min_pt=[-resolution//2+1, -resolution//2+1], max_pt=[resolution//2, resolution//2], shape=[resolution, resolution], dtype='float32',impl='numpy')

# Angles: uniformly spaced, n = 1000, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, samples)

# Detector: uniformly sampled, n = 500, min = -30, max = 30
detector_partition = odl.uniform_partition(-resolution+1, resolution, resolution*2)

# Make a parallel beam geometry with flat detector
geometry = odl.tomo.Parallel2dGeometry(angle_partition,detector_partition)
# geometry = odl.tomo.parallel_beam_geometry(reco_space, samples, det_shape=resolution*2)

# Ray transform (= forward projection).
radon = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# s = odl.tomo.backends.astra_cuda.astra_cuda_bp_scaling_factor(radon.range,reco_space,geometry)
# Fourier transform in detector direction
fourier = odl.trafos.FourierTransform(radon.range, axes=[1],impl='numpy')
# Create ramp in the detector 'direction'
ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))
# Create ramp filter via the convolution formula with fourier transforms
ramp_filter = fourier.inverse * ramp_function * (fourier)

# Create filtered back-projection by composing the back-projection (adjoint)
# with the ramp filter.
fbp = radon.adjoint * ramp_filter
model_fbp = OperatorModule(fbp)

lin = np.linspace(-1,1,resolution)
XX, YY = np.meshgrid(lin,lin)
circle = ((XX**2+YY**2)<=1)*1.

# # TESTs
# data_CT = data_CT_test
# files = os.listdir(data_CT)
# label = np.zeros((3,resolution,resolution))
# for idx in tqdm(range(3)): # len(files)
#     file = data_CT+os.sep+files[idx]
#     print(imageio.imread(file).max())
#     label[idx] = (imageio.imread(file)/255)*circle
# label = torch.from_numpy(label[:3]).type(torch_type).to(device)
# data = OperatorFunction.apply(radon, label).data
# SNR = np.repeat(np.random.rand(data.shape[0],1)*(SNR_max-SNR_min)+SNR_min,samples,1)
# nref = torch.std(data,(2))**2
# sigma_noise = torch.sqrt(torch.tensor(10**(-SNR/10)).to(device) * nref)
# proj = data + torch.randn_like(data)*sigma_noise[:,:,None]
# proj[:,:,:32] = 0
# proj[:,:,-32:] = 0
# recon = model_fbp(proj.view(-1, samples, proj.shape[2]))


# i = 1
# plt.figure(1)
# plt.clf()
# plt.imshow(label[i].detach().cpu())
# plt.colorbar()

# plt.figure(2)
# plt.clf()
# plt.imshow(proj[i].detach().cpu())
# plt.colorbar()

# plt.figure(3)
# plt.clf()
# plt.imshow(recon[i].detach().cpu())
# plt.colorbar()


# i = 1
# plt.figure(1)
# plt.clf()
# plt.imshow(label[i])
# plt.colorbar()

# plt.figure(2)
# plt.clf()
# plt.imshow(proj[i])
# plt.colorbar()

# plt.figure(3)
# plt.clf()
# plt.imshow(recon[i])
# plt.colorbar()


# idx = 100 
# data_CT = data_CT_train
# files = os.listdir(data_CT)
# file = data_CT+os.sep+files[idx]
# label = (imageio.imread(file)/255)*circle
# label = torch.from_numpy(label).type(torch_type).to(device)
# data = OperatorFunction.apply(radon, label).data
# # SNR = np.random.rand(data.shape[0])*(SNR_max-SNR_min)+SNR_min
# SNR = -10
# nref = torch.std(data,1)**2
# sigma_noise = torch.sqrt(torch.tensor(10**(-SNR/10)).to(device) * nref)
# proj = data + torch.randn_like(data)*sigma_noise[:,None]
# recon = model_fbp(proj.view(-1, samples, resolution))

# Train dataset
data_CT = data_CT_train
save_dir = "data/limited-CT/data_png_train_out"
save_dir_proj = "data/limited-CT/data_png_train_proj"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(save_dir_proj):
    os.makedirs(save_dir_proj)

files = os.listdir(data_CT)

label_np = np.zeros((len(files),resolution,resolution))
for idx in tqdm(range(len(files))): # len(files)
    file = data_CT+os.sep+files[idx]
    label_np[idx] = (imageio.imread(file)/255)*circle

label_tot = []
proj_tot = []
recon_tot = []
np.arange(0,label_np.shape[0],4)
for i in range(0,label_np.shape[0],label_np.shape[0]//4):
    print(i)
    label = torch.from_numpy(label_np[i:i+label_np.shape[0]//4]).type(torch_type).to(device)
    data = OperatorFunction.apply(radon, label).data
    SNR = np.repeat(np.random.rand(data.shape[0],1)*(SNR_max-SNR_min)+SNR_min,samples,1)
    nref = torch.std(data,(2))**2
    sigma_noise = torch.sqrt(torch.tensor(10**(-SNR/10)).to(device) * nref)
    proj = data + torch.randn_like(data)*sigma_noise[:,:,None]
    proj[:,:,:resolution//2] = 0
    proj[:,:,-resolution//2:] = 0
    recon = model_fbp(proj.view(-1, samples, proj.shape[2]))

    label_tot.append(label.detach().cpu())
    proj_tot.append(proj.detach().cpu())
    recon_tot.append(recon.detach().cpu())

label = np.zeros((len(files),resolution,resolution))
proj = np.zeros((len(files),samples,resolution*2))
recon = np.zeros((len(files),resolution,resolution))
for k, i in enumerate(range(0,label_np.shape[0],label_np.shape[0]//4)):
    label[i:i+label_np.shape[0]//4] = label_tot[k]
    proj[i:i+label_np.shape[0]//4] = proj_tot[k]
    recon[i:i+label_np.shape[0]//4] = recon_tot[k]
np.savez("data/limited-CT/data_train.npz",label=label,proj=proj,recon=recon)

for idx in tqdm(range(len(files))): # len(files)
    file = data_CT+os.sep+files[idx]
    out = recon[idx]
    out = (out-out.min())/(out.max()-out.min())
    imageio.imwrite(save_dir+os.sep+files[idx],np.clip(out*255,0,255).astype(np.uint8))
    out = proj[idx]
    out = (out-out.min())/(out.max()-out.min())
    imageio.imwrite(save_dir_proj+os.sep+files[idx],np.clip(out*255,0,255).astype(np.uint8))


# # Test dataset
data_CT = data_CT_test
save_dir = "data/limited-CT/data_png_test_out"
save_dir_proj = "data/limited-CT/data_png_test_proj"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(save_dir_proj):
    os.makedirs(save_dir_proj)
files = os.listdir(data_CT)

label = np.zeros((len(files),resolution,resolution))
for idx in tqdm(range(len(files))): # len(files)
    file = data_CT+os.sep+files[idx]
    label[idx] = (imageio.imread(file)/255)*circle

label = torch.from_numpy(label).type(torch_type).to(device)
data = OperatorFunction.apply(radon, label).data
SNR = np.repeat(np.random.rand(data.shape[0],1)*(SNR_max-SNR_min)+SNR_min,samples,1)
nref = torch.std(data,(2))**2
sigma_noise = torch.sqrt(torch.tensor(10**(-SNR/10)).to(device) * nref)
proj = data + torch.randn_like(data)*sigma_noise[:,:,None]
proj[:,:,:resolution//2] = 0
proj[:,:,-resolution//2:] = 0
recon = model_fbp(proj.view(-1, samples, proj.shape[2]))

np.savez("data/limited-CT/data_test.npz",label=label.detach().cpu().numpy(),proj=proj.detach().cpu().numpy(),recon=recon.detach().cpu().numpy())

for idx in tqdm(range(len(files))): # len(files)
    file = data_CT+os.sep+files[idx]
    out = recon[idx].detach().cpu().numpy()
    out = (out-out.min())/(out.max()-out.min())
    imageio.imwrite(save_dir+os.sep+files[idx],np.clip(out*255,0,255).astype(np.uint8))
    out = proj[idx].detach().cpu().numpy()
    out = (out-out.min())/(out.max()-out.min())
    imageio.imwrite(save_dir_proj+os.sep+files[idx],np.clip(out*255,0,255).astype(np.uint8))

