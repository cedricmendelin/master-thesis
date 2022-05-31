import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import matplotlib.pyplot as plt
from utils.ImageHelper import *
from utils.ODLHelperCustom import setup_forward_and_backward
from utils.ODLHelper import OperatorFunction, OperatorModule
from utils.SNRHelper import add_noise_to_sinograms, find_SNR

import os
import torch
import wandb
from tqdm import tqdm

np.random.seed(2022)
torch.manual_seed(2022)

test_data = "src/data/limited-CT/data_png_test/"
validation_files = os.listdir(test_data)
RESOLUTION = 64
validation_count = 100
N = 1024

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--snr", type=int, default=0)
args = parser.parse_args()

snr = args.snr


c = {
  "RESOLUTION" : RESOLUTION,
  "N" : N,
  "SNR" : snr,
  "Count" : validation_count,
}

wandb.init(project="FBP Validation LoDoPaB small", entity="cedric-mendelin", config=c)
wandb.run.name = f"FBP_{RESOLUTION}_{N}_{snr}"

x_validation = load_images_files_rescaled(test_data, validation_files, RESOLUTION, RESOLUTION, number=validation_count, num_seed=5, circle_padding=True)
t_validation_images = torch.from_numpy(x_validation).type(torch.float)

radon, fbp, pad = setup_forward_and_backward(RESOLUTION, N)

sinos = OperatorFunction.apply(radon, t_validation_images).data
noisy_sinos = add_noise_to_sinograms(sinos, snr)


for i in tqdm(range(validation_count)):

  loss_sino_noisy = torch.linalg.norm(noisy_sinos[i] - sinos[i])
  reco_noisy = fbp(pad(noisy_sinos[i])).data
  loss_reco_noisy = torch.linalg.norm(torch.from_numpy(reco_noisy) - t_validation_images[i])

  wandb.log({
    "val_idx" : i,
    "val_loss_sino_noisy" : loss_sino_noisy,
    "val_snr_sino_noisy" : find_SNR(sinos[i],  noisy_sinos[i]),
    "val_loss_reco_denoised" : loss_reco_noisy,
    "val_loss_reco_noisy" : loss_reco_noisy,
    "val_snr_reco_denoised" : find_SNR(t_validation_images[i],  torch.from_numpy(reco_noisy) ),
    "val_snr_reco_noisy" : find_SNR(t_validation_images[i],  torch.from_numpy(reco_noisy)),
    "val_reco_denoised" : wandb.Image(reco_noisy, caption="Reconstruction FBP"),
    "val_reco_noisy" : wandb.Image(reco_noisy, caption="Reconstruction noisy"),
    "val_clean" : wandb.Image(x_validation[i], caption="Original object")
  })


wandb.finish()
