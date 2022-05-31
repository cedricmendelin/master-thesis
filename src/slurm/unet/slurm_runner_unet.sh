#!/bin/bash

for snr in {0, -5, -10, -15} 
do 
  sbatch UNET_LoDoPaB_small.sh $snr
done

