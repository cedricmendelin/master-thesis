#!/bin/bash

#SBATCH --job-name=train_bunny        #Name of your job
#SBATCH --cpus-per-task=4           #Number of cores to reserve
#SBATCH --mem-per-cpu=12G           #Amount of RAM/core to reserve
#SBATCH --time=24:00:00             #Maximum allocated time
#SBATCH --qos=1day                 #Selected queue to allocate your job
#SBATCH --partition=pascal
#SBATCH --gres=gpu:3
#SBATCH --output=log/toyimages_out.%j   #Path and name to the file for the STDOUT
#SBATCH --error=log/toyimages_error_out.%j    #Path and name to the file for the STDERR

ml Python/3.6.6-fosscuda-2018b

source ~/venv_cu102/bin/activate
cd ~/master-thesis-git-nobackup

wandb offline

python ./src/GATDenoiserFixedImages_runner.py  \
        --graph_size 1024 \
        --samples 1024 \
        --resolution 100 \
        --image_path "data/bunnies/" \
        --validation_image_count 40 \
        --append_validation_images 10 \
        --use_wandb \
        --wandb_project "bunnies scicore" \
        --verbose \
        --save_model \
        --k_nn 8 \
        --validation_snrs 10 \
        --epochs 2000    \
        --gat_heads 16 \
        --gat_layers 4 \
        --gat_dropout 0.03 \
        --gat_weight_decay 0.0005 \
        --gat_learning_rate 0.01 \
        --gat_snr_lower 10 \
        --gat_snr_upper 10 \
        --batch_size 320 \
        --loss "SINO" \
