#!/bin/bash

#SBATCH --job-name=train_gat_mendel #Name of your job
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

python ./src/GATDenoiser_runner.py  \
        --samples 1024 \
        --resolution 196 \
        --image_path "src/data/val2017/" \
        --input_image_count 2000 \
        --validation_image_count 200 \
        --append_validation_images 50 \
        --use_wandb True \
        --wandb_project "cocoimages scicore" \
        --verbose True \
        --debug_plots False \
        --save_model True \
        --k_nn 2 \
        --add_circle_padding False \
        --validation_snrs -5 2 10 25 \
        --epochs 2000    \
        --gat_heads 4 \
        --gat_layers 3 \
        --gat_dropout 0.03 \
        --gat_weight_decay 0.0005 \
        --gat_learning_rate 0.01 \
