import os

from utils.GATDenoiserHelper import GatBase
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import matplotlib.pyplot as plt
from utils.ImageHelper import *

from utils.ODLHelper import OperatorFunction, OperatorModule
from utils.SNRHelper import add_noise_to_sinograms, find_SNR

from utils.Plotting import plot_imshow
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
N = 1024

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--graph_size", type=int, default=1024)
parser.add_argument("--samples", type=int, default=1024)
parser.add_argument("--resolution", type=int, default=64)

parser.add_argument("--use_wandb", action='store_true', default=False)
parser.add_argument("--gat_layers", type=int, default=4)
parser.add_argument("--gat_heads", type=int, default=8)
parser.add_argument("--gat_dropout", type=float, default=0.05)
parser.add_argument("--gat_weight_decay", type=float, default=0.0005)
parser.add_argument("--gat_learning_rate", type=float, default=0.01)
parser.add_argument("--gat_use_conv", action='store_true', default=False)
parser.add_argument("--gat_conv_kernel", type=int, default=3)
parser.add_argument("--gat_conv_padding", type=int, default=1)
parser.add_argument("--gat_conv_N_latent", type=int, default=1)
parser.add_argument("--unet_refinement", action='store_true', default=False)
parser.add_argument("--unet_path", type=str, default="models/unet.pt")

parser.add_argument("--validation_snrs", nargs="+", type=int, default=[0])
parser.add_argument("--validation_image_count", type=int, default=100)

parser.add_argument("--k_nn", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=256)

parser.add_argument("--verbose", action='store_true', default=False)

parser.add_argument("--model_state_path", type=str, default="denoiser/small_experiments/test_gat.pt")
parser.add_argument("--run_name", type=str, default="Blub Validation")
parser.add_argument("--wandb_project", type=str, default="Some Test Validation")
args = parser.parse_args()

validation_count = args.validation_image_count
x_validation = load_images_files_rescaled(test_data, validation_files, RESOLUTION, RESOLUTION, number=validation_count, num_seed=5, circle_padding=True)

model_state = torch.load(args.model_state_path)
validator = GatBase.create_validator(args, model_state, run_name=args.run_name)

validator.validate(x_validation, args.validation_snrs, args.batch_size)
validator.finish_wandb()

