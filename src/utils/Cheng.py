from vedo import dataurl, Mesh, mesh2Volume, volumeFromMesh

from tqdm import tqdm
#from skimage.transform import rotate
from scipy.spatial.transform import Rotation as R
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
#import torch.nn.functional as F
import networkx as nx
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import  rescale, radon, iradon
from skimage.morphology import disk
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import numpy as np
#from FBB import FBBasis2D
from scipy import sparse
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from skimage.transform import rotate
from scipy.spatial.transform import Rotation as R
import sys
#import plotly.graph_objects as go
sys.path.insert(0, '..')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch

from scipy import sparse
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from skimage.transform import rotate
import sys


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch


# Helpers:

def rotation3d(d3_points,phi,psi,theta):
    r = R.from_euler('xyz', [phi, psi, theta], degrees=True)
    return r.apply(d3_points)

def inplane_distance_points(dataset,M=100):
    print('search one')
    N=dataset.shape[0]
    data2d=torch.tensor(dataset,dtype=torch.float32)


    theta=torch.arange(0,2*np.pi,2*np.pi/M)
    rot_matrix=torch.empty(M,2,2)
    rot_matrix[:,0,0]=torch.cos(theta)
    rot_matrix[:,0,1]=-torch.sin(theta)
    rot_matrix[:,1,0]=torch.sin(theta)
    rot_matrix[:,1,1]=torch.cos(theta)

    distance_matrix=torch.empty((N,N))
    angle_matrix=torch.empty((N,N))
    for i in tqdm(range(N)):
        pts2dI=data2d[i,:,:]
        pts2dI_bt=torch.broadcast_to(pts2dI,(data2d.shape[0],M,data2d.shape[1],data2d.shape[2]))
        pts2d_rot=torch.einsum('nzp,mpq->nmzq', data2d, rot_matrix)
        diff=torch.norm(pts2d_rot-pts2dI_bt,dim=(2,3))
        diff,min_index=torch.min(diff,dim=1)
        distance_matrix[i,:]=diff
        angle_matrix[i]=theta[min_index]
    return distance_matrix.numpy(),angle_matrix.numpy()


def get_img(d3_points,resolution=50,xlim=[-1,1],ylim=[-1,1],theta=None,block=False):
    [xlim_l,xlim_u]=xlim
    [ylim_l,ylim_u]=ylim
    devx=(xlim_u-xlim_l)/resolution
    devy=(ylim_u-ylim_l)/resolution
    data_2d=d3_points[:,:2]
    if theta is not None:
        rot_matrix=np.empty((2,2))
        rot_matrix[0,0]=np.cos(theta)
        rot_matrix[0,1]=-np.sin(theta)
        rot_matrix[1,0]=np.sin(theta)
        rot_matrix[1,1]=np.cos(theta)
        data_2d=data_2d@rot_matrix
    data_2d=data_2d.T


    xcord=(data_2d[0,:]-(xlim_l))/devx
    ycord=(data_2d[1,:]-(ylim_l))/devy


    xcord[xcord>=resolution]=resolution-1
    xcord[ycord>=resolution]=resolution-1
    xcord[xcord<0]=0
    ycord[ycord<0]=0


    xcord=torch.tensor(xcord,dtype=torch.long)
    ycord=torch.tensor(ycord,dtype=torch.long)
    z=torch.ones_like(xcord,dtype=torch.int)
    img=torch.zeros((resolution,resolution),dtype=torch.int)
    img.index_put_((xcord, ycord), z, accumulate=True)
    img=img.numpy()
    if block:
        img[img>0]=1
    return img

def get_img_3d(d3_points,resolution=50,xlim=[-1,1],ylim=[-1,1],zlim=[-1,1],theta=None,block=False):
    [xlim_l,xlim_u]=xlim
    [ylim_l,ylim_u]=ylim
    [zlim_l,zlim_u]=zlim
    devx=(xlim_u-xlim_l)/resolution
    devy=(ylim_u-ylim_l)/resolution
    devz=(ylim_u-ylim_l)/resolution
    data_3d=d3_points.T



    xcord=(data_3d[0,:]-(xlim_l))/devx
    ycord=(data_3d[1,:]-(ylim_l))/devy
    zcord=(data_3d[2,:]-(zlim_l))/devz



    xcord[xcord>=resolution]=resolution-1
    xcord[ycord>=resolution]=resolution-1
    zcord[zcord>=resolution]=resolution-1
    xcord[xcord<0]=0
    ycord[ycord<0]=0
    zcord[zcord<0]=0


    xcord=torch.tensor(xcord,dtype=torch.long)
    ycord=torch.tensor(ycord,dtype=torch.long)
    zcord=torch.tensor(zcord,dtype=torch.long)
    z=torch.ones_like(xcord,dtype=torch.int)
    img=torch.zeros((resolution,resolution,resolution),dtype=torch.int)
    img.index_put_((xcord, ycord,zcord), z, accumulate=True)
    img=img.numpy()
    if block:
        img[img>0]=1
    return img


def getChengBunny(resolution=50):
  pts=np.load("src/maps/bunny.npy")
  pts=pts[np.random.permutation(pts.shape[0]),:]


  pts[:,0]=pts[:,0]-(pts[:,0].max()+pts[:,0].min())/2
  pts[:,1]=pts[:,1]-(pts[:,1].max()+pts[:,1].min())/2
  pts[:,2]=pts[:,2]-(pts[:,2].max()+pts[:,2].min())/2

  print(pts.shape)
  pts_max=np.linalg.norm(pts,axis=1).max()
  pts=pts/pts_max

  print(pts.shape)

  return get_img_3d(pts,resolution=resolution,block=True).astype(np.float64)

from .Normalization import normalize_min_max
from .Geometry import downsample_voxels

def getVedoBunny(resolution=64, padding=7):
  mesh = Mesh(dataurl + "bunny.obj").normalize().subdivide()

  vol = mesh2Volume(mesh, spacing=(0.01,0.01,0.01)).tonumpy()
  V = normalize_min_max(downsample_voxels(vol, resolution - (2*padding)))

  #237, 234, 183
  p = (padding, padding)
  V = np.pad(V, (p, p, p) , mode='constant', constant_values=0).astype(np.float64)
  
  return V

def getVedoBunny2(resolution=64, padding=7):
  mesh = Mesh(dataurl + "bunny.obj").normalize().subdivide()

  vol = volumeFromMesh(mesh, dims=(resolution, resolution, resolution)).tonumpy()
  V = normalize_min_max(downsample_voxels(vol, resolution - (2*padding)))

  #237, 234, 183
  p = (padding, padding)
  V = np.pad(V, (p, p, p) , mode='constant', constant_values=0).astype(np.float64)
  
  return V

import mrcfile   

def getMapFile(map_file="emd_25792.map", resolution=64):
  # load map file
  vol =  mrcfile.open('src/maps/' + map_file).data.astype(np.float64)

  vol = normalize_min_max(vol)
  
  threshold = 0.35
  vol[vol < threshold] = 0

  V = normalize_min_max(downsample_voxels(vol, resolution))
  return V