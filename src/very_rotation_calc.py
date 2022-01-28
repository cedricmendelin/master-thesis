from math import degrees
import numpy as np
from utils.Geometry import *
from utils.Graph import *
from utils.AspireHelpers import *
from utils.Data import *
from utils.Plotting import *
import os.path as path
from os import mkdir
import gemmi
import vedo
import mrcfile

n = 1000
normal_dist_sphere = sampling_sphere(n)
  
fibo_sphere =  fibonacci_sphere(n)

G_aligned = GAlign("embedding", emb1=normal_dist_sphere, emb2=fibo_sphere).get_align()

embedding = fibo_sphere[G_aligned]
G_aligned_2 = GAlign("embedding", emb1=normal_dist_sphere, emb2=embedding).get_align()
print(G_aligned_2)

for i in range(n):
  if i % 100 == 0:
    print(f"Actual {normal_dist_sphere[i]}, estimated {embedding[i]}")
  

dist = np.linalg.norm(normal_dist_sphere - embedding)
print(dist/n)


rotations = calc_rotation_from_points_on_sphere_2(embedding)
origin = np.array([1,0,0])
rotated_p = R.from_euler(rotations[1]).apply(origin)






input("enter to terminate")
