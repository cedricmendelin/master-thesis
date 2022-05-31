#!/bin/bash

for k in {2,6,8,10,20} 
do 
  for g in {512,1024,2048} 
  do
    sbatch lodopab_small_knn_graph.sh $k $g
  done 
done

