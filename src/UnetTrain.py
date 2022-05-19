
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
    torch.cuda.set_device(2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_type=torch.float

data_CT_test = "data/limited-CT/data_png_test"
data_CT_train = "data/limited-CT/data_png_train"
resolution = 64
samples = 1024
SNR_min = -20
SNR_max = 0
save_dir = "results/Unet"
train = False

batch_size = 32
lr = 1e-3
epochs = 10000

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


#######################################
### Prepare dataset
#######################################
def find_SNR(ref, x):
    dif = torch.sum((ref-x)**2)
    nref = torch.sum(ref**2)
    return 10 * torch.log10((nref+1e-16)/(dif+1e-16))

def find_sigma_noise(SNR_value, x_ref):
    nref = torch.mean(x_ref**2)
    sigma_noise = (10**(-SNR_value/10)) * nref
    return torch.sqrt(sigma_noise)

def add_noise_np(SNR, sinogram):
    nref = np.mean(sinogram**2)
    sigma_noise = (10**(-SNR/10)) * nref
    sigma = np.sqrt(sigma_noise)
    print("noise sigma:", sigma)
    noise = np.random.randn(sinogram.shape[0], sinogram.shape[1]) * sigma
    noisy_sino = sinogram + noise
    return noisy_sino

# class MyDataset(Dataset):
#     def __init__(self, data_CT):
#         self.data_CT = data_CT
#         self.files = os.listdir(data_CT)

#         reco_space = odl.uniform_discr(
#             min_pt=[-20, -20], max_pt=[20, 20], shape=[resolution, resolution], dtype='float32')

#         # Angles: uniformly spaced, n = 1000, min = 0, max = pi
#         angle_partition = odl.uniform_partition(0, np.pi, samples)

#         # Detector: uniformly sampled, n = 500, min = -30, max = 30
#         detector_partition = odl.uniform_partition(-30, 30, resolution)

#         # Make a parallel beam geometry with flat detector
#         geometry = odl.tomo.Parallel2dGeometry(angle_partition,detector_partition)

#         # Ray transform (= forward projection).
#         self.radon = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

#         # Fourier transform in detector direction
#         fourier = odl.trafos.FourierTransform(self.radon.range, axes=[1])
#         # Create ramp in the detector direction
#         ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))
#         # Create ramp filter via the convolution formula with fourier transforms
#         ramp_filter = fourier.inverse * ramp_function * fourier

#         # Create filtered back-projection by composing the back-projection (adjoint)
#         # with the ramp filter.
#         self.fbp = self.radon.adjoint * ramp_filter

#         lin = np.linspace(-1,1,resolution)
#         XX, YY = np.meshgrid(lin,lin)
#         self.circle = ((XX**2+YY**2)<=1)*1.

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
        
#         label = (imageio.imread(self.data_CT+os.sep+self.files[idx])/255)*self.circle
#         label = torch.from_numpy(label)
#         data = OperatorFunction.apply(self.radon, label).data
#         # add noise
#         SNR = np.random.rand()*(SNR_max-SNR_min)+SNR_min
#         sigma = find_sigma_noise(SNR, data)
#         data = data + torch.randn_like(data)*sigma

#         # reconstruct
#         data = OperatorFunction.apply(self.fbp,data.view(1, samples, resolution))

#         sample = {'data': data.view(resolution,resolution).type(torch_type), 'label': label.type(torch_type)}
#         return sample


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx,:], 'label': self.label[idx]}
        return sample

lin = np.linspace(-1,1,resolution)
XX, YY = np.meshgrid(lin,lin)
circle = ((XX**2+YY**2)<=1)*1.

# Train dataset
data_CT = "data/limited-CT/data_png_train"
data_CT_out = "data/limited-CT/data_png_train_out"
files = os.listdir(data_CT)
label_train = np.zeros((len(files),resolution,resolution))
data_train = np.zeros((len(files),resolution,resolution))
for idx in range(len(files)): 
    file = data_CT+os.sep+files[idx]
    label_train[idx] = (imageio.imread(file)/255)*circle
    file = data_CT_out+os.sep+files[idx]
    data_train[idx] = (imageio.imread(file)/255)*circle
label_train = torch.tensor(label_train).type(torch_type).to(device)
data_train = torch.tensor(data_train).type(torch_type).to(device)

data_CT = "data/limited-CT/data_png_test"
data_CT_out = "data/limited-CT/data_png_test_out"
files = os.listdir(data_CT)
label_test = np.zeros((len(files),resolution,resolution))
data_test = np.zeros((len(files),resolution,resolution))
for idx in range(len(files)): 
    file = data_CT+os.sep+files[idx]
    label_test[idx] = (imageio.imread(file)/255)*circle
    file = data_CT_out+os.sep+files[idx]
    data_test[idx] = (imageio.imread(file)/255)*circle
label_test = torch.tensor(label_test).type(torch_type).to(device)
data_test = torch.tensor(data_test).type(torch_type).to(device)


train_loader = DataLoader(dataset=MyDataset(data_train,label_train), 
                                        batch_size=batch_size, 
                                        shuffle=True)
test_loader = DataLoader(dataset=MyDataset(data_test,label_test), 
                                        batch_size=batch_size, 
                                        shuffle=True)



# data_CT = data_CT_test
# files = os.listdir(data_CT)
# reco_space = odl.uniform_discr(
#     min_pt=[-20, -20], max_pt=[20, 20], shape=[resolution, resolution], dtype='float32')

# # Angles: uniformly spaced, n = 1000, min = 0, max = pi
# angle_partition = odl.uniform_partition(0, np.pi, samples)

# # Detector: uniformly sampled, n = 500, min = -30, max = 30
# detector_partition = odl.uniform_partition(-30, 30, resolution)

# # Make a parallel beam geometry with flat detector
# geometry = odl.tomo.Parallel2dGeometry(angle_partition,detector_partition)

# # Ray transform (= forward projection).
# radon = odl.tomo.RayTransform(reco_space, geometry)

# # Fourier transform in detector direction
# fourier = odl.trafos.FourierTransform(radon.range, axes=[1])
# # Create ramp in the detector direction
# ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))
# # Create ramp filter via the convolution formula with fourier transforms
# ramp_filter = fourier.inverse * ramp_function * fourier

# # Create filtered back-projection by composing the back-projection (adjoint)
# # with the ramp filter.
# fbp = radon.adjoint * ramp_filter

# lin = np.linspace(-1,1,resolution)
# XX, YY = np.meshgrid(lin,lin)
# circle = ((XX**2+YY**2)<=1)*1.

# label = torch.zeros((len(files),resolution,resolution)).type(torch_type).to(device)
# for idx in range(len(files)):
#     label[idx] = torch.from_numpy((imageio.imread(data_CT_test+os.sep+files[idx])/255)*circle)
#     data = OperatorFunction.apply(radon, label).data

#     # add noise
#     SNR = np.random.rand()*(SNR_max-SNR_min)+SNR_min
#     sigma = find_sigma_noise(SNR, data)
#     data = data + torch.randn_like(data)*sigma

# # reconstruct
# data = OperatorFunction.apply(self.fbp,data.view(1, samples, resolution))





#######################################
### Define network
#######################################
net = models.UNet(nfilter=128).to(device).train()
net.summary()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-6)

#######################################
### Define network
#######################################
loss_tot = []
# Load previously trained model
# if os.path.exists('net.pt'):
#     print("### Load previous model")
#     checkpoint = torch.load("net.pt",map_location=device)
#     net.load_state_dict(checkpoint['model_state_dict']) 
#     loss_tot = checkpoint['loss_tot']

Ndisp = 500
if train:
    for ep in range(epochs):
        for i, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
            if ep%1==0 and ep!=0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr*(1-(1-0.1)*ep/epochs)

            input = data['data'].to(device)
            label = data['label'].to(device)

            optimizer.zero_grad()
            y_est = net(input.view(-1,1,resolution,resolution))
            
            loss = criterion(y_est.view(-1,resolution,resolution),label)

            loss.backward()
            optimizer.step()

            loss_tot.append(loss.item())

            if i%Ndisp==0 and i!=0:
                print("{0}/{1} -- Loss over last {2} iter: {3}".format(ep,epochs,Ndisp,np.mean(loss_tot[-Ndisp:])))

                out = y_est[0,0].detach().cpu().numpy()
                inp = input[0].detach().cpu().numpy()
                tru = label[0].detach().cpu().numpy()
                out = (out-out.min())/(out.max()-out.min())
                inp = (inp-inp.min())/(inp.max()-inp.min())
                tru = (tru-tru.min())/(tru.max()-tru.min())
                imageio.imwrite(save_dir+os.sep+"ep_"+str(ep)+"_i_"+str(i)+"_out.png", np.clip(out*255,0,255).astype(np.uint8) )
                imageio.imwrite(save_dir+os.sep+"ep_"+str(ep)+"_i_"+str(i)+"_input.png", np.clip(inp*255,0,255).astype(np.uint8) )
                imageio.imwrite(save_dir+os.sep+"ep_"+str(ep)+"_i_"+str(i)+"_true.png", np.clip(tru*255,0,255).astype(np.uint8) )

                torch.save({
                    'ep': ep,
                    'i': i,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_tot': loss_tot,
                    }, "net.pt")

else:
    save_dir_test = save_dir+os.sep+"test"
    if not os.path.exists(save_dir_test):
        os.makedirs(save_dir_test)
    checkpoint = torch.load("net.pt",map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    cpt = 0 
    for i, data in enumerate(test_loader):
        input = data['data'].to(device)
        label = data['label'].to(device)
        y_est = net(input.view(-1,1,resolution,resolution))
        loss = criterion(y_est.view(-1,resolution,resolution),label)

        print("{0} -- Loss: {1}".format(i,loss.item() ))

        for k in range(y_est.shape[0]):
            out = y_est[k,0].detach().cpu().numpy()
            inp = input[k].detach().cpu().numpy()
            tru = label[k].detach().cpu().numpy()
            out = (out-out.min())/(out.max()-out.min())
            inp = (inp-inp.min())/(inp.max()-inp.min())
            tru = (tru-tru.min())/(tru.max()-tru.min())
            imageio.imwrite(save_dir_test+os.sep+str(cpt)+"_out.png", np.clip(out*255,0,255).astype(np.uint8) )
            imageio.imwrite(save_dir_test+os.sep+str(cpt)+"_input.png", np.clip(inp*255,0,255).astype(np.uint8) )
            imageio.imwrite(save_dir_test+os.sep+str(cpt)+"_true.png", np.clip(tru*255,0,255).astype(np.uint8) )
            cpt += 1