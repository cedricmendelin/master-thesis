import time
import argparse

import os
import torch
import pickle
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

matplotlib.use('TkAgg')
if(not plt.isinteractive):
    plt.ion()
    
from skimage.transform import resize
from utils.helper import normalize, clip_to_uint8

from utils.generateHelper import *


import os
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.laplacianHelper import *
from scipy.spatial import distance_matrix


from tqdm import tqdm


# import warnings
# warnings.simplefilter('default')


#Running Parameters
generateVoxel=True
generateProjection=True
computeLaplacian=True
plotEmbedding=True

#Script Parameters
modelName="FLATFOOT_StanfordBunny_jmil_HIGH_RES_Smoothed.stl"
voxelSize=0.02
size=100

N=50
sigmaPSF=0
SNR=5
Ntheta=1000
dofAngle=4
Nz=10
K=6
epsilon = 1
shiftWidth=0
cuda=True
verbose = True
save_dir = 'results'

voxelmodelName=modelName[:-4]+"_VOXELSZ_"+str(voxelSize)+"_SZ_"+str(size)
projectionName=voxelmodelName+"_n_"+str(N)+"_psf_"+str(sigmaPSF)+"_SNR_"+str(SNR)+"_Ntheta_"+str(Ntheta)+"_DAngle_"+str(dofAngle)+"_shiftW_"+str(shiftWidth)
embeddingName=projectionName+'_K_'+str(K)

##################################################################

np.random.seed(0)
torch_type=torch.float
use_cuda=torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print("GPU: ", use_cuda)

## General parameters
n = N#100 # number of pixel in each direction
sigma_psf =  sigmaPSF #1 # std of the Gaussian PSF (pixel)
SNR_value = SNR #100 # wanted SNR value
SUBPLOTSIZE = 5
fileName = modelName+'_n_'+str(n)+'_psf_'+str(sigma_psf)+'_SNR_'+str(SNR_value)+'_Ntheta_'+str(Ntheta)+'_DAngle_'+str(dofAngle)+'_shiftW_'+str(shiftWidth)
print(fileName)

if not os.path.exists(save_dir+os.sep+"data_generation"+os.sep+modelName):
    os.makedirs(save_dir+os.sep+"data_generation"+os.sep+modelName)

#Generate Angles and shifts
thetaSet = generate_rotation(Ntheta,dofAngle)
Ntheta = thetaSet.shape[0]
V=np.load('./toyModel/'+voxelmodelName+'.npy')
V=resize(V,(n,n,n))
g,_,paddingSize = getPSF(sigma_psf,n)
shifts = np.random.random(size=(Ntheta,2))*shiftWidth


# If possible, use cuda and pytorch to fasten the data generatin
if use_cuda and cuda:
    # Convert numpy to cuda torch
    thetaSet_t = torch.tensor(thetaSet.copy()).type(torch_type).to(device)
    g_t = torch.tensor(g).type(torch_type).to(device)
    V_t = torch.FloatTensor(V).unsqueeze(0).type(torch_type).to(device)
    ## Generate an experiments
    P_t = torch.zeros((Ntheta,n,n)).type(torch_type).to(device)
    P_t_ = torch.zeros((Ntheta,n,n)).type(torch_type).to(device)
    angleMatrixTorch = getRotationMatrix_t(thetaSet_t,'XYZ').type(torch_type).to(device)
    # r = scipy_rot.from_euler('xyz', thetaSet, degrees=True)
    # mat_r = r.as_matrix()
    print('Generating Noisy Projections in Pytorch...')
    for k in range(Ntheta):
        if (k%50==49):
            print("{0}/{1}".format(k+1,Ntheta))
        P0 = forward_t(V_t, angleMatrixTorch[k], shifts[k],g_t,paddingSize)
        sigma_noise = find_sigma_noise_t(SNR_value,P0)
        y = P0 + torch.randn_like(P0)*sigma_noise
        P_t[k] = y
        P_t_[k] = P0
    P = P_t.detach().cpu().numpy()
    P_ = P_t_.detach().cpu().numpy()

else:
    ## Generate an experiments
    P = np.zeros((Ntheta,n,n))
    P_ = np.zeros((Ntheta,n,n))
    angleMatrix = getRotationMatrix(thetaSet,'XYZ')
    print('Generating Noisy Projections...')
    for k in range(Ntheta):
        if (k%50==49):
            print("{0}/{1}".format(k+1,Ntheta))
        P0 = forward(V, angleMatrix[k], shifts[k],g,paddingSize)
        sigma_noise = find_sigma_noise(SNR_value,P0)
        y = P0 + np.random.randn(n,n)*sigma_noise
        P[k] = y
        P_[k] = P0

x,y,z = sph2cart(thetaSet[:,0]/180*(np.pi), thetaSet[:,1]/180*(np.pi))
color_RGBA=np.empty((Ntheta,3))
color_RGBA[:,0]=x
color_RGBA[:,1]=y
color_RGBA[:,2]=z
color_RGBA = (color_RGBA-color_RGBA.min())/(color_RGBA.max()-color_RGBA.min())

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z,c=color_RGBA)

fig = plt.figure(2)
ax = fig.add_subplot(141)
ax.imshow(P_[0])
ax = fig.add_subplot(142)
ax.imshow(P_[3])
ax = fig.add_subplot(143)
ax.imshow(P_[6])
ax = fig.add_subplot(144)
ax.imshow(P_[8])


#####################################################################
if not os.path.exists("./embedding"):
    os.makedirs("./embedding")

#Laplacian Parameters 
fileName = projectionName

refl_bool = True
GL_invariant = False

## Load data
# n,sigma_psf,SNR_value,Ntheta,thetaSet,shifts,P,P_,dofAngle = np.load('./data/'+fileName+'.pkl',allow_pickle= True)
print(thetaSet.shape)
saveFilename = fileName+'_K_'+str(K)

X = P_.reshape(P_.shape[0],-1)
dist_str = 'l2'

# if dist_str == 'rotation':
#     dist, ind, _, refl = rotation_invariant_knn(X,K=K,ell_max=None,refl_bool=refl_bool)
#     neigbourMatrix = np.zeros((X.shape[0],X.shape[0]))
#     distMatrix = np.zeros((X.shape[0],X.shape[0]))
#     for k in range(X.shape[0]):
#         neigbourMatrix[k,ind[k,:]] = 1
#         distMatrix[k,ind[k,:]] = dist[k]
#         neigbourMatrix[ind[k,:],k] = 1
#         distMatrix[ind[k,:],k] = dist[k]
#     affinMatrix = np.exp(-distMatrix**2/epsilon)*neigbourMatrix
#     print('Distance Computed')
# elif dist_str == 'angle':
#     dist = distance_matrix(X,X)
#     dist = np.minimum(dist,360-dist)
# else:
#     dist = distance_matrix(X,X)
if dofAngle==4:
    tmp = P_.reshape(Ntheta//Nz,Nz,n,n)
    dist=np.zeros((Ntheta//Nz,Ntheta//Nz))
    for i in tqdm(range(Ntheta//Nz)):
        for j in range(Ntheta//Nz):
            # dist[i]=np.linalg.norm(tmp[i:i+1,0:1] - tmp,axis=2).min(1)
            d_tmp = np.zeros((Nz,1))
            for k in range(Nz):
                    d_tmp[k] = np.linalg.norm(tmp[i,0] - tmp[j,k])
            dist[i,j]=d_tmp.min(0)

    dist /= dist.max() # to make epsilon almost adimensional
if dofAngle==3:
    tmp = P_.reshape(Ntheta//Nz,Nz,n,n)
    dist_, ind, _, refl = rotation_invariant_knn(tmp[:,0],K=K+1,ell_max=None,refl_bool=False)
    neigbourMatrix = np.zeros((tmp.shape[0],tmp.shape[0]))
    distMatrix = np.zeros((tmp.shape[0],tmp.shape[0]))
    for k in range(tmp.shape[0]):
        neigbourMatrix[k,ind[k,:]] = 1
        distMatrix[k,ind[k,:]] = dist_[k]
        distMatrix[k,k] = 0
        # neigbourMatrix[ind[k,:],k] = 1
        # distMatrix[ind[k,:],k] = dist_[k]
    affinMatrix = np.exp(-distMatrix**2/epsilon)*neigbourMatrix
    print('Distance Computed')
    dist2 = distMatrix
    dist2/=dist2.max()

    from scipy.ndimage import rotate
    from aspire.basis import ffb_2d
    fb = ffb_2d.FFBBasis2D((tmp[0].shape[1],tmp[0].shape[1]),ell_max=None,dtype=np.float64)

    tmp = P_.reshape(Ntheta//Nz,Nz,n,n)
    dist3 = np.zeros((Ntheta//Nz,Ntheta//Nz))
    for i in tqdm(range(Ntheta//Nz)):
        im1 = tmp[i,0]
        for j in range(Ntheta//Nz):
            im2 = tmp[j,0]
            # c2 = fb.evaluate_t(im2)
            dd = np.zeros(10)
            for k in range(10):
                im2_rot = rotate(im2,k*360/10,reshape=False)
                # im2_rot = fb.evaluate(fb.rotate(c2,k*2*np.pi/10)).data[0]
                dd[k] = np.linalg.norm(im1-im2_rot)
            dist3[i,j] = dd.min()

    tmp = P_.reshape(Ntheta//Nz,Nz,n,n)
    dist4 = np.zeros((Ntheta//Nz,Ntheta//Nz))
    for i in tqdm(range(Ntheta//Nz)):
        im1 = tmp[i,0]
        for j in range(Ntheta//Nz):
            im2 = tmp[j,0]
            c2 = fb.evaluate_t(im2.astype(np.float64)).astype(np.float64)
            dd = np.zeros(10)
            for k in range(10):
                # im2_rot = rotate(im2,k*360/10,reshape=False)
                im2_rot = fb.evaluate(fb.rotate(c2,k*2*np.pi/10)).data[0]
                dd[k] = np.linalg.norm(im1-im2_rot)
            dist4[i,j] = dd.min()
            



# if dist_str != 'rotation':
#     print('Distance Computed')
#     neigbourMatrix = generateNeigbourHoodGraph(dist,K)
#     neigbourMatrix = (((neigbourMatrix + neigbourMatrix.T)>0)*1).astype('float64')
#     affinMatrix = np.exp(-dist**2/epsilon)

def neighbotFromX(X,n_neighbors=10):
    A=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[0]):
        diss_list=X[i]
        neb_list=np.argsort(diss_list)[1:n_neighbors+1]
        A[i][neb_list]=1
    return A

from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from scipy.sparse.linalg import eigsh
data=dist.reshape(dist.shape[0],-1)
A_nl1=neighbotFromX(data,n_neighbors=K)
# A_nl2=(dist2>1e-7)*1
# data=dist3.reshape(dist3.shape[0],-1)
# A_nl3=neighbotFromX(data,n_neighbors=K)

A_nl = A_nl1

# A_k=sparse.csr_matrix(A_nl)
# A_knn = 0.5*(A_k + sparse.csr_matrix.transpose(A_k))
# L = sparse.csr_matrix(sparse.diags(np.ones(A_nl.shape[0]))) - A_knn
A_knn=0.5*(A_nl+A_nl.T)
A_knn[A_knn>1]=1
D=np.diag(A_knn.sum(axis=1))
L = np.diag(A_knn.sum(axis=1)) - A_knn
# D_invdiv2=np.diag(np.sqrt(1/A_knn.sum(axis=1)))
# L=np.eye(A_knn.shape[0])-D_invdiv2@A_knn@D_invdiv2
L=sparse.csr_matrix(L)
eigenValues, eigenVectors=eigsh(L,k=4,which='SM')
idx = np.argsort(eigenValues)
Phi0 = eigenVectors[:,1:]

if dofAngle==4:
    tt = thetaSet.reshape(Ntheta//Nz,Nz,3)
    x,y,z = sph2cart(tt[:,0,0]/180*(np.pi),tt[:,0,1]/180*(np.pi))
    color_RGBA=np.empty((tt.shape[0],3))
else:
    x,y,z = sph2cart(thetaSet[:,0]/180*(np.pi),thetaSet[:,1]/180*(np.pi))
    color_RGBA=np.empty((thetaSet.shape[0],3))
color_RGBA[:,0]=x
color_RGBA[:,1]=y
color_RGBA[:,2]=z
color_RGBA = (color_RGBA-color_RGBA.min())/(color_RGBA.max()-color_RGBA.min())


fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Phi0[:,0], Phi0[:,1], Phi0[:,2],c=color_RGBA)
    
plt.show()


