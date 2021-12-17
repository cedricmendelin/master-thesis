import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def diffusion_map(X=None, alpha=0.15):
  """
  1. Compute the euclidean_distances matrix: dists
  2. Compute the kernel matrix K=exp(-dists^2/alpha)
  3. Scale the row of the kernel matrix P=D^{-1}K

  """
  dists = euclidean_distances(X,X)
  K = np.exp(-dists**2/alpha)
  D = np.sum(K,axis=0)
  P = np.diag(1/D) @ K 
  return P

def diffusion_distance(P, n_eign=2,t=1):
  """
  Your task: given diffusion_map: P, time of diffuion: t, and number of eignvalue you need: n_eign
  
  compute (1) coordinate in low dimension: psi  (2) diffusion distance matrix: D
  

  1. Compute eigenValues and  eigenVectors of P
  2. Select the largest n_eign number of eigenValues and corresponding eigenVectors.
  3. Compute the coordinate psi_i=[eigenValues_1^t eigenVectors_1[i],eigenValues_2^t eigenVectors_2[i]...]
  4. Compute the euclidean_distances matrix: D=euclidean_distances(psi, psi)
  """
  eValues, eVectors = np.linalg.eig(P)
  print(eValues)
  eValues=eValues.real
  eVectors=eVectors.real
  eValueIndexOrder = np.argsort(-np.abs(eValues))
  #print(eValues[eValueIndexOrder])
  eValuesSorted = np.real(eValues[eValueIndexOrder[1:n_eign+1]])
  eVectorsSorted = np.real(eVectors[:,eValueIndexOrder[1:n_eign+1]])
  eValuesSortedT = eValuesSorted**t
  psi   = eVectorsSorted @ np.diag(eValuesSortedT)

  return euclidean_distances(psi,psi) ,psi