import torch

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

# some small test

# N = 100
# M = 65
# snr= 10

# x_ref = torch.randn(N, M) * 5

# print(torch.max(x_ref))
# print(torch.min(x_ref))
# x = add_noise(snr, x_ref)

# snr_calc = find_SNR(x_ref, x)

# print(f"SNR: calculated: {snr_calc} , should be arround {snr} ")

