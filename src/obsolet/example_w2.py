import numpy as np
from ot import emd2 #https://pythonot.github.io/
import matplotlib.pyplot as plt; plt.ion()
from scipy.spatial import distance_matrix

N = 256 # number of points

## 2D example 
pts1 = np.random.randn(N,2)
pts1 /= np.linalg.norm(pts1,axis=1,keepdims=True) # points on the sphere
lin = np.linspace(0,2*np.pi,N)
pts2 = np.array([np.cos(lin),np.sin(lin)]).T

plt.figure(1) 
plt.scatter(pts1[:,0],pts1[:,1],marker='x',c=np.arctan2(pts1[:,0],pts1[:,1]),label='pts1')
plt.scatter(pts2[:,0],pts2[:,1],marker='o',c=np.arctan2(pts2[:,0],pts2[:,1]),label='pts2: target')

dist = distance_matrix(pts1,pts2) # l2 distance between set of points
ones_ = np.ones(N)/N # weights of the distributions. Each points as the same waight
W, log = emd2(ones_.copy(order='C'),ones_.copy(order='C'),dist.copy(order='C'),return_matrix=True)
tr_plan = log['G'] # Contains the transport pla, i.e. tr_plan[i,j] is how many of points i we should move to point j. 
# tr_plan[i,:] will contain exactly one value non-zero, unless to points in pts2 are equally space from one point oin pts1.


_, ind_to_move = np.where(tr_plan)
pts1_new = pts2[ind_to_move] # new assignement of points in pts1
plt.scatter(0.9*pts1_new[:,0],0.9*pts1_new[:,1],marker='+',c=np.arctan2(pts1[:,0],pts1[:,1]),label='new pts1') # *0.9 to see better
plt.legend()


def sampling_sphere(Ntheta):
    th = np.random.random(Ntheta) * np.pi * 2
    x = np.random.random(Ntheta) * 2 - 1
    out = np.array([np.cos(th) * np.sqrt(1 - x**2), np.sin(th) * np.sqrt(1 - x**2),x]).T
    return out


## 3D example 
pts1 = np.random.randn(N,3)
pts1 /= np.linalg.norm(pts1,axis=1,keepdims=True) # points on the sphere
pts2 = sampling_sphere(N)

fig = plt.figure(2) 
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts1[:,0],pts1[:,1],pts1[:,2],marker='x',c=np.arctan2(pts1[:,0],pts1[:,1]),label='pts1')
ax.scatter(pts2[:,0],pts2[:,1],pts2[:,2],marker='o',c=np.arctan2(pts2[:,0],pts2[:,1]),label='pts2: target')

dist = distance_matrix(pts1,pts2) # l2 distance between set of points
ones_ = np.ones(N)/N # weights of the distributions. Each points as the same waight
W, log = emd2(ones_.copy(order='C'),ones_.copy(order='C'),dist.copy(order='C'),return_matrix=True)
tr_plan = log['G'] # Contains the transport pla, i.e. tr_plan[i,j] is how many of points i we should move to point j. 
# tr_plan[i,:] will contain exactly one value non-zero, unless to points in pts2 are equally space from one point oin pts1.


_, ind_to_move = np.where(tr_plan)
pts1_new = pts2[ind_to_move] # new assignement of points in pts1
ax.scatter(0.9*pts1_new[:,0],0.9*pts1_new[:,1],0.9*pts1_new[:,2],marker='+',c=np.arctan2(pts1[:,0],pts1[:,1]),label='new pts1') # *0.9 to see better
ax.legend()

input("enter to terminate")