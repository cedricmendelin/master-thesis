import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from utils.Graph import *
plt.ion()


## Generate some toy dataset
N = 64 # Dimension of images
L=10 # numer of images​
x1 = np.random.randn(L,N,N)


# Get rotations and correlations with coefficients c1
M=180 # number of angles to try in the rotation invariant distance
K=4 # number of neighbors 

classes, reflections, rotations, shifts, correlations = aspire_knn_with_rotation_invariant_distance(x1, K)

print(classes)
print(reflections)
print(rotations)
print(shifts) # will be None
print(correlations)