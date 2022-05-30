import torch
import numpy as np

def add_noise(SNR, sinogram):
    sigma = find_sigma_noise(SNR, sinogram)
    noise = torch.randn(sinogram.shape[0], sinogram.shape[1]) * sigma
    noisy_sino = sinogram + noise
    return noisy_sino


def add_noise_to_sinograms(sinograms, snr):
    noisy_sinograms = torch.empty_like(sinograms)
    
    nref = torch.std(sinograms,(2))**2
    sigma_noise = torch.sqrt(torch.tensor(10**(-snr/10)) * nref)
    noisy_sinograms = sinograms + torch.randn_like(sinograms) * sigma_noise[:,:,None]

    return noisy_sinograms

def find_SNR(ref, x):
    dif = torch.std((ref-x))**2
    nref = torch.std(ref)**2
    return 10 * torch.log10((nref+1e-16)/(dif+1e-16))

def find_sigma_noise(SNR_value, x_ref):
    nref = torch.std(x_ref)**2
    sigma_noise = (10**(-SNR_value/10)) * nref
    return torch.sqrt(sigma_noise)

def add_noise_np(SNR, sinogram):
    nref = np.std(sinogram)**2
    sigma_noise = (10**(-SNR/10)) * nref
    sigma = np.sqrt(sigma_noise)

    noise = np.random.randn(sinogram.shape[0], sinogram.shape[1]) * sigma
    noisy_sino = sinogram + noise
    return noisy_sino


# some small test
# N = 1024
# M = 50
# x_ref = torch.randn(10, N, M) 
# snr = -5

# x = add_noise_to_sinograms(x_ref, snr)

# for i in range(10):
#     snr_calc = find_SNR(x_ref[i], x[i])
#     # old_method_x = add_noise(snr, x_ref[i])
#     # snr_calc2 = find_SNR(x_ref[i], old_method_x)
#     print(f"SNR: calculated: {snr_calc} , should be arround {snr} ")
#     # print(f"Old SNR: calculated: {snr_calc2} , should be arround {snr} ")








# x_ref = torch.randn(N, M) * 5

# print(torch.max(x_ref))
# print(torch.min(x_ref))
# x = add_noise(snr, x_ref)

# snr_calc = find_SNR(x_ref, x)

# print(f"SNR: calculated: {snr_calc} , should be arround {snr} ")
# import time
# N = 1024
# M = 1024
# snr= 5
# samples = 10

# x_refs  = torch.randn(samples, N, M)


# # add noise together:
# nref = torch.mean(x_refs ** 2, (1,2))
# sigmas = (10**(-snr/10)) * nref
# n = torch.randn((samples, N, M))
# # not working
# noise = torch.einsum('nij,s->sij', n, sigmas)
# print(n[0,0,0])
# print(sigmas[0])
# print(noise[0,0,0])
# noisy_sino = x_refs + noise

# for i in range(samples):
#     print(f"      SNR: calculated: {find_SNR(x_refs[i], noisy_sino[i])} , should be arround {snr} ")
# #     print(f"old - SNR: calculated: {find_SNR(x_refs[i], add_noise(snr, x_refs[i]))} , should be arround {snr} ")

