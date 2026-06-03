import numpy as np
import matplotlib.pyplot as pl
import torch
from psf import Convolution

pl.close('all')

config = {
        'n_pixel': 152,
        'n_pixel_out': 128,
        'npix_apodization': 12,
        'wavelength': [3934.0, 6173.0, 8542.0, 6563.0],
        'diameter': [100.0, 100.0, 100.0, 144.0],
        'pix_size': [0.038, 0.059, 0.059, 0.04979],
        'central_obs': [0.0, 0.0, 0.0, 40.0],
        'n_modes': 44,
        'r0_min': 3.0,
        'r0_max': 15.0,
        'mask_cutoff': 0.9
    }

conv = Convolution(config)
psf_diff = conv.gen_psf(r0=5.0, wl=2)

psf_diff = torch.tensor(psf_diff.astype('float32'))

npix = config['n_pixel']

n_seq = 1
nf = 1
jitter = torch.zeros((n_seq, nf, 2, 2))
jitter[0, 0, 0, :] = torch.tensor([-1.5, -0.6])
jitter[0, 0, 1, :] = torch.tensor([-0.3, -0.3])


K = 50
sigma_alias = 0.02
norm = 'ortho'

# Short frequency jitter
x = torch.linspace(-1, 1, npix)
y = torch.linspace(-1, 1, npix)
X = torch.tensor(x)
Y = torch.tensor(y)
X = X.view(1, 1, npix, 1)
Y = Y.view(1, 1, 1, npix)

        
# 2. Create the time parameter tau from 0 to 1
# Shape: (K, 1) to broadcast with (x, y) coordinates later
tau = torch.linspace(0, 1, K)

P1 = jitter[:, :, 0, :] # Shape: (n_seq, nf, 2)
P2 = jitter[:, :, 1, :] # Shape: (n_seq, nf, 2)

jitter_psf = torch.zeros((n_seq, nf, npix, npix))

for k in range(K):

    t = tau[k]

    # Compute curve point for step K using the quadratic Bézier formula
    p_k = 2 * (1 - t) * t * P1 + (t ** 2) * P2  # Shape [n_seq, nf, 2]

    # Split into x and y components
    p_x = p_k[..., 0].unsqueeze(-1).unsqueeze(-1)
    p_y = p_k[..., 1].unsqueeze(-1).unsqueeze(-1)

    # 4. Compute 1D separable distances
    # dist_x_sq shape: [n_seq, nf, nx, 1]
    # dist_y_sq shape: [n_seq, nf, 1, ny]
    dist_x_sq = (X - p_x) ** 2
    dist_y_sq = (Y - p_y) ** 2
    
    # 5. Compute 1D Gaussians
    psf_x = torch.exp(-dist_x_sq / (2 * sigma_alias ** 2))
    psf_y = torch.exp(-dist_y_sq / (2 * sigma_alias ** 2))

    # 6. Outer product (via broadcasting) creates the [nx, ny] 2D Gaussian
    # Add to the accumulator
    jitter_psf += (psf_x * psf_y)

jitter_psf = torch.fft.fftshift(jitter_psf, dim=(-2, -1)) # Shift zero frequency to center

# 8. Normalize to ensure energy conservation (sum to 1)
# Avoid division by zero with a tiny epsilon
psf_sum = torch.sum(jitter_psf, dim=(-1, -2), keepdim=True)
jitter_psf = jitter_psf / (psf_sum + 1e-8)

jitter_psf = jitter_psf[0, 0, ...] # Take the first frame for visualization

otf_diff = torch.fft.fft2(psf_diff, norm=norm)
otf_jitter = torch.fft.fft2(jitter_psf, norm=norm)

print(f"Diffractin PSF sum: {torch.sum(psf_diff)}")
print(f"Jitter PSF sum: {torch.sum(jitter_psf)}")

otf = otf_diff * otf_jitter
psf = torch.fft.ifft2(otf).real
print(f"Combined PSF sum: {torch.sum(psf)}")

print(f"Diffraction OTF peak value: {torch.max(torch.abs(otf))}")
print(f"Jitter OTF peak value: {torch.max(torch.abs(otf_jitter))}")
print(f"Combined OTF peak value: {torch.max(torch.abs(otf))}")

fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(15, 5))
im = ax[0].imshow(np.fft.fftshift(psf_diff.numpy()), extent=[-1, 1, -1, 1])
pl.colorbar(im, ax=ax[0])
ax[0].set_title('Diffraction')

im = ax[1].imshow(np.fft.fftshift(jitter_psf.numpy()), extent=[-1, 1, -1, 1])
pl.colorbar(im, ax=ax[1])
ax[1].set_title('Jitter')

im = ax[2].imshow(np.fft.fftshift(psf.numpy()), extent=[-1, 1, -1, 1])
pl.colorbar(im, ax=ax[2])
ax[2].set_title('Diffraction * Jitter')
        

# High-frequency jitter
# pl.close('all')

# sx = 1.0
# sy = 0.2
# rxy = 0.1

# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# X, Y = np.meshgrid(x, y)

# pix_size = x[1] - x[0]

# xy = (X/sx)**2 + (Y/sy)**2 - 2.0*rxy*X*Y/(sx*sy)
# psf = np.exp(-0.5 * xy / (1.0 - rxy**2))
# psf = psf / (2.0 * np.pi * sx * sy * np.sqrt(1.0 - rxy**2))

# psf = np.fft.fftshift(psf)

# psf_f = np.fft.fft2(psf, norm='ortho')

# f_x = np.fft.fftfreq(psf.shape[0], d=(x[1] - x[0]))
# f_y = np.fft.fftfreq(psf.shape[1], d=(y[1] - y[0]))
# F_X, F_Y = np.meshgrid(f_x, f_y)

# otf = np.exp(-2.0 * np.pi**2 * (F_X**2 * sx**2 + F_Y**2 * sy**2 + 2.0*rxy*F_X*F_Y*sx*sy))

# fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(15, 10))

# im = ax[0, 0].imshow(psf_f.real)
# pl.colorbar(im, ax=ax[0, 0])
# ax[0, 0].set_title('PSF Fourier Transform (Real Part)')

# im = ax[0, 1].imshow(np.fft.fftshift(np.fft.ifft2(psf_f, norm='ortho').real), extent=(x[0], x[-1], y[0], y[-1]))
# pl.colorbar(im, ax=ax[0, 1])
# ax[0, 1].set_title('Inverse FFT of PSF Fourier Transform (Real Part)')

# im = ax[0, 2].imshow(np.fft.fftshift(psf), extent=(x[0], x[-1], y[0], y[-1]))
# pl.colorbar(im, ax=ax[0, 2])

# im = ax[1, 0].imshow(otf)
# pl.colorbar(im, ax=ax[1, 0])
# ax[1, 0].set_title('OTF')

# im = ax[1, 1].imshow(np.fft.fftshift(np.fft.ifft2(otf, norm='ortho').real), extent=(x[0], x[-1], y[0], y[-1]))
# pl.colorbar(im, ax=ax[1, 1])
# ax[1, 1].set_title('Inverse FFT of OTF (Real Part)')

# area_psf = np.sum(psf) * pix_size**2
# area_psf_recovered = np.sum(np.fft.ifft2(psf_f, norm='ortho').real) * pix_size**2

# area_psf2 = np.sum(np.fft.ifft2(otf, norm='ortho').real) * pix_size**2

# print(f"Area of PSF original: {area_psf}")
# print(f"Area of PSF recovered from Inverse Fourier Transform: {area_psf_recovered}")
# print(f"Area of PSF recovered from OTF: {area_psf2}")