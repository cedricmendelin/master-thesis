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
import bm3d
import time
import wandb
from tqdm import tqdm

torch.manual_seed(2022)

test_data = "src/data/limited-CT/data_png_test/"
validation_files = os.listdir(test_data)

RESOLUTION = 64
validation_count = 3553
N = 1024
snr = 0

# bm3d_before_fbp:
# True: bm3d on sinogram, 
# False, bm3d on noisy sinogram
from enum import Enum
class BM3DType(Enum):
    """ Enum class for different types of losses.
        Sino : l2-distance between clean sinogram and denoised sinogram.
        Fbp: l2.distance between clean input images and fbp(denoised sinogram).
    """
    SINO = 0,
    RECO = 1,
    BOTH=2,


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bm3d', type=str, default='RECO',
                    choices=[i.name.upper() for i in BM3DType])
args = parser.parse_args()

bm3d_type = BM3DType[args.bm3d.upper()]

c = {
  "RESOLUTION" : RESOLUTION,
  "N" : N,
  "SNR" : snr,
  "Count" : validation_count,
  "Type": bm3d_type,
}

wandb.init(project="BM3D Validation LoDoPaB large", entity="cedric-mendelin", config=c)
wandb.run.name = f"BM3D_{RESOLUTION}_{N}_{snr}_{N}_{bm3d_type}"


x_validation = load_images_files_rescaled(test_data, validation_files, RESOLUTION, RESOLUTION, number=validation_count, num_seed=5, circle_padding=True)
t_validation_images = torch.from_numpy(x_validation).type(torch.float)

radon, fbp = setup_forward_and_backward(RESOLUTION, N)
sinos = OperatorFunction.apply(radon, t_validation_images).data
noisy_sinos = add_noise_to_sinograms(sinos, snr)

for i in tqdm(range(validation_count)):
  
  # denoise sinogram
  if bm3d_type == BM3DType.BOTH or bm3d_type == BM3DType.SINO:
    std = torch.std(sinos[i] - noisy_sinos[i])
    sino_bm3d = bm3d.bm3d(noisy_sinos[i], sigma_psd=std, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    sino_bm3d = torch.from_numpy(sino_bm3d)
  else: 
    sino_bm3d = noisy_sinos[i]


  loss_sino_bm3d = torch.linalg.norm(sino_bm3d - sinos[i])
  loss_sino_noisy = torch.linalg.norm(noisy_sinos[i] - sinos[i])

  reco_noisy = fbp(noisy_sinos[i]).data

  # denoise reco
  if bm3d_type == BM3DType.RECO:
    std2 = torch.std(t_validation_images[i] - reco_noisy)
    reco_bm3d = bm3d.bm3d(reco_noisy, sigma_psd=std2, stage_arg=bm3d.BM3DStages.ALL_STAGES)
  elif bm3d_type == BM3DType.SINO:
    # reco on denoised sino
    reco_bm3d = fbp(sino_bm3d).data
  elif bm3d_type == BM3DType.BOTH:
    # fbp with denoised sino
    reco_bm3d = fbp(sino_bm3d).data
    std2 = torch.std(t_validation_images[i] - reco_bm3d)
    # refinement, 2nd time bm3d
    reco_bm3d = bm3d.bm3d(reco_noisy, sigma_psd=std2, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    

  loss_reco_bm3d = torch.linalg.norm(torch.from_numpy(reco_bm3d) - t_validation_images[i])
  loss_reco_noisy = torch.linalg.norm(torch.from_numpy(reco_noisy) - t_validation_images[i])

  wandb.log({
    "val_idx" : i,
    "val_loss_sino_denoised" : loss_sino_bm3d,
    "val_loss_sino_noisy" : loss_sino_noisy,
    "val_snr_sino_denoised" : find_SNR(sinos[i],  sino_bm3d),
    "val_snr_sino_noisy" : find_SNR(sinos[i],  noisy_sinos[i]),
    "val_loss_reco_denoised" : loss_reco_bm3d,
    "val_loss_reco_noisy" : loss_reco_noisy,
    "val_snr_reco_denoised" : find_SNR(t_validation_images[i],  torch.from_numpy(reco_bm3d)),
    "val_snr_reco_noisy" : find_SNR(t_validation_images[i],  torch.from_numpy(reco_noisy)),
    "val_reco_denoised" : wandb.Image(reco_bm3d, caption="Reconstruction BM3D"),
    "val_reco_noisy" : wandb.Image(reco_noisy, caption="Reconstruction noisy"),
    "val_clean" : wandb.Image(x_validation[i], caption="Original object")
  })


wandb.finish()
