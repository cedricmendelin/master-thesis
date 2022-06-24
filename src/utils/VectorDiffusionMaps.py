import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
# import tqdm

#def vector_diffusion_map():

#def vector_diffusion_distance(P, n_eign=2, t=1):


def calculate_S(N, K, clean_graph, epsilon_vdm, dim = 2, sym = True):
  # O = np.zeros((N, K, dim, dim))
  # W = np.zeros((N, K, dim))
  # s_weights = np.zeros((N, K, dim, dim))
  S = np.zeros((N * dim, N * dim))
  D = np.zeros((N * dim, N * dim))
  A_s = np.zeros((N,N))
  degree = np.zeros(N)

  # for all images
  # calculate S, which includes W and O
  for i in range(N):
    d_i = 0

    # iterate over all neighbours
    for j  in range(K):
      if clean_graph.classes[i,j] == i:
        A_s[i,i] = 1
        continue

      # optimal alignment angle and rotation
      theta = clean_graph.angles[i,j] /180*np.pi
      c, s = np.cos(theta), np.sin(theta)
      o_ij = np.array(((c, -s), (s, c)))

      # weights including gaussian kernel
      # Equation 2.6
      w_ij = np.exp(- clean_graph.distance[i,j] ** 2 / epsilon_vdm)
      # w_ij = 1

      # weighted alignment
      # Equation 3.1
      # s_weights[i,j] = o_ij * w_ij

      # assign values to S
      d_i = d_i + w_ij # Equation 3.3
      row = int(i * dim)
      col = int(clean_graph.classes[i,j] * dim)
      
      S[row:row + dim, col : col+dim] = o_ij * w_ij
      A_s[i, clean_graph.classes[i,j]] = 1
      
      if sym:
        S[col : col+dim, row:row + dim] = o_ij.T * w_ij
        A_s[clean_graph.classes[i,j], i] = 1
      

    # set values of D
    row = i * dim
    D[row:row+dim, row:row+dim] = d_i * np.identity(dim) # Equation 3.2
    degree[i] = d_i

  return S, D, A_s


def calculate_vdm_affinity(S_tilde_t2, N, dim=2,):
  affinity = np.zeros((N,N))
  for i in range(N):
    for j in range(i + 1, N):
      ii = i*dim
      jj = j*dim
      affinity[i,j] = np.linalg.norm(S_tilde_t2[ii:ii+dim, jj:jj+dim], ord='fro') ** 2
      affinity[j,i] = np.linalg.norm(S_tilde_t2[ii:ii+dim, jj:jj+dim], ord='fro') ** 2
      
  return affinity

def calculate_vdm_distance(hs_norm, N):
  vdm_distance = np.zeros((N,N))
  for i in range(N):
    for j in range(N):
      vdm_distance[i,j] = hs_norm[i, i] + hs_norm[j, j] - (2 * hs_norm[i, j])
  return vdm_distance

from scipy.linalg import fractional_matrix_power
def verfiy_spectral_decomposition(eign_values, eign_vecs, eign_value_idx, S_tilde, t_start = 1, t_stop=11, t_step = 1, n_eign_start=2, n_eign_stop=22, n_eign_step=2):
  dim_t = math.ceil((t_stop - t_start) / t_step)
  dim_eign = math.ceil((n_eign_stop - n_eign_start) / n_eign_step)
  #n_eign_start=10, n_eign_stop=54, n_eign_step=4)
  
  plot_result_s_tilde = np.zeros((dim_t, dim_eign))
  plot_result_s_tilde_2t = np.zeros_like(plot_result_s_tilde)

  print(plot_result_s_tilde.shape)
  for t in range(t_start, t_stop, t_step):
    for n_eign in range(n_eign_start, n_eign_stop, n_eign_step):
      t_idx = math.floor((t - t_start) / t_step) 
      n_eign_idx = math.floor((n_eign - n_eign_start) / n_eign_step)
      
      print(t, n_eign, t_idx, n_eign_idx)

      t_2 = 2 * t

      S_tilde_t2 = fractional_matrix_power(S_tilde, t_2)

      eValuesSorted = eign_values[eign_value_idx[0:n_eign+1]]
      eVectorsSorted = eign_vecs[:,eign_value_idx[0:n_eign+1]]

      S_tilde_decomposed_truncated = (eVectorsSorted[:,None,:]*eVectorsSorted[None,:,:]*eValuesSorted[None,None,:]).sum(2)
      S_tilde_t2_decomposed_truncated = (eVectorsSorted[:,None,:]*eVectorsSorted[None,:,:]*eValuesSorted[None,None,:] ** t_2).sum(2)

      diff_s_tilde = np.linalg.norm(S_tilde - S_tilde_decomposed_truncated) / np.linalg.norm(S_tilde)
      diff_s_2t_tilde = np.linalg.norm(S_tilde_t2 - S_tilde_t2_decomposed_truncated) / np.linalg.norm(S_tilde_t2)

      plot_result_s_tilde[t_idx][n_eign_idx] = diff_s_tilde
      plot_result_s_tilde_2t[t_idx][n_eign_idx] = diff_s_2t_tilde

  # plt.plot(plot_result_s_tilde)
  print("debug plot:")
  print(plot_result_s_tilde_2t[:,:])
  print(plot_result_s_tilde_2t)
  
  plt.plot(range(n_eign_start, n_eign_stop, n_eign_step), plot_result_s_tilde_2t.T)
  plt.legend(range(t_start, t_stop, t_step))
  plt.xlabel("Number of evals")
  plt.ylabel("Relative Diff between original and decomposed version")
  plt.xticks(range(n_eign_start, n_eign_stop, n_eign_step))
  plt.title(f"Spectal decompositio S_tilde power 2t")
  plt.show()

import math
def verify_epsilon(distances, epsilon_vdm_list, show=False):
  number_of_plots = len(epsilon_vdm_list)
  dimy = math.ceil(number_of_plots / 2)
  dimx = 2
  fig, axs = plt.subplots(dimy, dimx)

  for i in range(number_of_plots):
    epsilon_vdm = epsilon_vdm_list[i]
    #aspire_angles_xyz = np.array([R.from_euler("ZYZ", angles).as_euler("XYZ") for angles in sim.angles])
    x = np.array([ dist ** 2 / epsilon_vdm for dist in distances])
    y = np.array([ np.exp( - x_i) for x_i in x])
    axs[math.floor(i/2), i % 2].scatter(x,y)
    axs[math.floor(i/2), i % 2].set_title(f"Epsilon VDM {epsilon_vdm}, dist mean: {distances.mean()}")
    axs[math.floor(i/2), i % 2].set_yticks(np.arange(0, 1.1, 0.2))
  if show:
    plt.show()
