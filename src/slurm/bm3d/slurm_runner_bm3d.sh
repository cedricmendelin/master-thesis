#!/bin/bash

for bm3d in {"RECO", "SINO"} 
do 
  for snr in {0, -5, -10, -15} 
  do
    sbatch BM3D_LoDoPaB_small.sh $bm3d $snr
  done 
done

