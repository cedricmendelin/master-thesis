import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import matplotlib.pyplot as plt
from utils.ImageHelper import *
from utils.ODLHelperCustom import setup_forward_and_backward
from utils.ODLHelper import OperatorFunction
from utils.SNRHelper import add_noise_to_sinograms, find_SNR


import os
import torch
from utils.UNetModel import UNet
import wandb
from tqdm import tqdm

np.random.seed(2022)
torch.manual_seed(2022)

test_data = "src/data/limited-CT/data_png_test/"
validation_files = os.listdir(test_data)

RESOLUTION = 64
validation_count = 100
N = 1024
snr = 0

c = {
  "RESOLUTION" : RESOLUTION,
  "N" : N,
  "SNR" : snr,
  "Count" : validation_count,
}

wandb.init(project="U-Net Validation LoDoPaB small", entity="cedric-mendelin", config=c)
wandb.run.name = f"UNET_{RESOLUTION}_{N}_{snr}_{N}"


x_validation = load_images_files_rescaled(test_data, validation_files, RESOLUTION, RESOLUTION, number=validation_count, num_seed=5, circle_padding=True)
t_validation_images = torch.from_numpy(x_validation).type(torch.float)

radon, fbp = setup_forward_and_backward(RESOLUTION, N)
sinos = OperatorFunction.apply(radon, t_validation_images).data
noisy_sinos = add_noise_to_sinograms(sinos, snr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("models/unet_128.pt", map_location=device)
unet = UNet(nfilter=128).eval()
unet.load_state_dict(checkpoint['model_state_dict'])    


for i in tqdm(range(validation_count)):

  loss_sino_noisy = torch.linalg.norm(noisy_sinos[i] - sinos[i])
  reco_noisy = fbp(noisy_sinos[i]).data
  print(reco_noisy.shape)
  print(torch.from_numpy(reco_noisy).view(-1, 1, RESOLUTION, RESOLUTION).size())
  reco_fbp_unet = unet(torch.from_numpy(reco_noisy).view(-1, 1, RESOLUTION, RESOLUTION) ).view(RESOLUTION, RESOLUTION).cpu().detach()
  
 
  loss_reco_unet = torch.linalg.norm(reco_fbp_unet - t_validation_images[i])
  loss_reco_noisy = torch.linalg.norm(torch.from_numpy(reco_noisy) - t_validation_images[i])

  # print(loss_reco_unet)
  # print(reco_fbp_unet.size())
  # print(find_SNR(t_validation_images[i],  reco_fbp_unet))

  wandb.log({
    "val_idx" : i,
    "val_loss_sino_noisy" : loss_sino_noisy,
    "val_snr_sino_noisy" : find_SNR(sinos[i],  noisy_sinos[i]),
    "val_loss_reco_denoised" : loss_reco_unet,
    "val_loss_reco_noisy" : loss_reco_noisy,
    "val_snr_reco_denoised" : find_SNR(t_validation_images[i],  reco_fbp_unet),
    "val_snr_reco_noisy" : find_SNR(t_validation_images[i],  torch.from_numpy(reco_noisy)),
    "val_reco_denoised" : wandb.Image(reco_fbp_unet.numpy(), caption="Reconstruction U-Net"),
    "val_reco_noisy" : wandb.Image(reco_noisy, caption="Reconstruction noisy"),
    "val_clean" : wandb.Image(x_validation[i], caption="Original object")
  })


wandb.finish()
