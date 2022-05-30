#!/bin/bash

#SBATCH --job-name=lodopab_small #Name of your job
#SBATCH --cpus-per-task=4           #Number of cores to reserve
#SBATCH --mem-per-cpu=12G           #Amount of RAM/core to reserve
#SBATCH --time=24:00:00             #Maximum allocated time
#SBATCH --qos=1day                 #Selected queue to allocate your job
#SBATCH --partition=pascal
#SBATCH --gres=gpu:3
#SBATCH --output=log/lodopab_small.%j   #Path and name to the file for the STDOUT
#SBATCH --error=log/lodopab_small_error.%j    #Path and name to the file for the STDERR

ml Python/3.6.6-fosscuda-2018b

source ~/venv_cu102/bin/activate
cd ~/master-thesis-git-nobackup

wandb offline

python ./src/LoDoPaB_small_experiment_runner.py  \
        --samples 1024 \
        --resolution 64 \
        --image_path "src/data/limited-CT/data_png_train/" \
        --validation_image_path "src/data/limited-CT/data_png_test/" \
        --add_circle_padding \
        --validation_image_count 100 \
        --use_wandb \
        --verbose \
        --save_model \
        --validation_snrs 0 \
        --epochs 200    \
        --gat_heads 1 \
        --gat_layers 3 \
        --gat_dropout 0.03 \
        --gat_weight_decay 0.0005 \
        --gat_learning_rate 0.01 \
        --gat_snr_lower 0 \
        --gat_snr_upper 0 \
        --batch_size 64 \
        --loss "FBP" \
        --unet_refinement \
        --gat_use_conv \
        --gat_conv_kernel 3 \
        --gat_conv_padding 1 \
        --gat_conv_N_latent 1 \
        --unet_path "models/unet.pt" \
        --wandb_project "LoDoPaB Small Test" \
        --model_dir "denoiser/small_experiments/" \
        --graph_size 1024 \
        --k_nn 8 \
        --run_name "test" 
                
