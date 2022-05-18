#!/bin/bash

#SBATCH --job-name=traing_coco #Name of your job
#SBATCH --cpus-per-task=4           #Number of cores to reserve
#SBATCH --mem-per-cpu=12G           #Amount of RAM/core to reserve
#SBATCH --time=24:00:00             #Maximum allocated time
#SBATCH --qos=1day                 #Selected queue to allocate your job
#SBATCH --partition=pascal
#SBATCH --gres=gpu:3
#SBATCH --output=log/cocoimages_out.%j   #Path and name to the file for the STDOUT
#SBATCH --error=log/cocoimages_error_out.%j    #Path and name to the file for the STDERR

ml Python/3.6.6-fosscuda-2018b

source ~/venv_cu102/bin/activate
cd ~/master-thesis-git-nobackup

wandb offline

python ./src/GATDenoiserFixedImages_runner.py  \
        --graph_size 1024 \
        --samples 8192 \
        --resolution 64 \
        --image_path "src/data/limited-CT/data_png_train/" \
        --validation_image_path "src/data/limited-CT/data_png_test/" \
        --validation_image_count 100 \
        --use_wandb True \
        --verbose True \
        --save_model True \
        --k_nn 8 \
        --validation_snrs 0 \
        --epochs 200    \
        --gat_heads 8 \
        --gat_layers 4 \
        --gat_dropout 0.03 \
        --gat_weight_decay 0.0005 \
        --gat_learning_rate 0.01 \
        --gat_snr_lower 0 \
        --gat_snr_upper 0 \
        --batch_size 64 \
        --loss "FBP" \
        --unet_refinement True \
        --gat_use_conv True \
        --gat_conv_kernel 3 \
        --gat_conv_padding 1 \
        --gat_conv_N_latent \
        --unet_path "models/unet.pt" \
        --wandb_project "LoDoPaB-CT"
