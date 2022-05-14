import torch
import numpy as np

def add_noise(SNR, sinogram):
    sigma = find_sigma_noise(SNR, sinogram)
    noise = torch.randn(sinogram.shape[0], sinogram.shape[1]) * sigma
    noisy_sino = sinogram + noise
    return noisy_sino


def add_noise_to_sinograms(sinograms, snr):
    noisy_sinograms = torch.empty_like(sinograms)
    for i in range(sinograms.shape[0]):
        noisy_sinograms[i] = add_noise(snr, sinograms[i])

    return noisy_sinograms

def find_SNR(ref, x):
    dif = torch.sum((ref-x)**2)
    nref = torch.sum(ref**2)
    return 10 * torch.log10((nref+1e-16)/(dif+1e-16))

def find_sigma_noise(SNR_value, x_ref):
    nref = torch.mean(x_ref**2)
    sigma_noise = (10**(-SNR_value/10)) * nref
    return torch.sqrt(sigma_noise)

def add_noise_np(SNR, sinogram):
    nref = np.mean(sinogram**2)
    sigma_noise = (10**(-SNR/10)) * nref
    sigma = np.sqrt(sigma_noise)
    print("noise sigma:", sigma)
    noise = np.random.randn(sinogram.shape[0], sinogram.shape[1]) * sigma
    noisy_sino = sinogram + noise
    return noisy_sino


# some small test



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

