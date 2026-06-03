import numpy as np
import matplotlib.pyplot as pl
import torch
from convolution import Convolution
from tqdm import tqdm
import torchmfbd

def generate_image_with_aapsd(size, D, wavelength, pixel_size, K, v0, beta, mean_image):
    """
    Generates an image with a power spectral density proportional to 1/f^beta.
    size: tuple (width, height)
    beta: exponent (0 = white noise, 1 = pink noise, 2 = brown noise)
    """

    cutoff = D / (wavelength * 1e-8) / 206265.0


    w, h = size
    # 1. Create White Noise
    white_noise = np.random.standard_normal(size)
    
    # 2. Fourier Transform
    noise_fft = np.fft.fft2(white_noise, norm='ortho')
    
    # 3. Create a radial frequency grid
    # Get coordinates relative to the center
    u = np.fft.fftfreq(w, d=pixel_size) / cutoff
    v = np.fft.fftfreq(h, d=pixel_size) / cutoff
    u, v = np.meshgrid(u, v)
    
    # Calculate radial frequency f = sqrt(u^2 + v^2)
    rho = np.sqrt(u**2 + v**2)    
    
    # 4. Generate the filter (Square root because PSD is power, we need amplitude)
    # Power Spectral Density P(f) ~ 1 / f^beta
    # Amplitude A(f) ~ sqrt(P(f)) ~ 1 / f^(beta/2)
    psd = K / (1.0 + (rho/v0)**2)**(beta / 2.0)
    
    f_filter = np.sqrt(psd)
    
    # 5. Apply filter and Inverse FFT
    filtered_fft = noise_fft * f_filter
    filtered_fft[rho == 0.0] = np.sqrt(w*h) * mean_image # Set DC component to 0

    img_back = np.fft.ifft2(filtered_fft, norm='ortho')
    
    # Return real part and normalize
    result = np.real(img_back)
    return result, f_filter, rho


if __name__ == "__main__":
    pl.close('all')
    show = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration file
    config = {
            'n_pixel': 256,
            'wavelength': 8542,
            'apodization': 12,
            'diameter': 100.0,
            'pix_size': 0.059,
            'central_obs': 0.0,
            'basis': 'zernike',
            'n_modes': np.sum(np.arange(20)+2),
        }
    
    conv = Convolution(config)
    
    # Generate artificial image with a power spectral density
    D = config['diameter']
    wavelength = config['wavelength']
    pixel_size = config['pix_size']
    npix = config['n_pixel']
    K_orig = 100.0
    v0_orig = 0.1
    beta_orig = 2.5
    mean_image = 1.0
    noise = 1e-2
    cutoff = D / (wavelength * 1e-8) / 206265.0

    img, sqrt_psd, rho = generate_image_with_aapsd((npix, npix), D=D, wavelength=wavelength, pixel_size=pixel_size, K=K_orig, v0=v0_orig, beta=beta_orig, mean_image=mean_image)

    img = torch.tensor(img, dtype=torch.float32).to(device)
    rho = torch.tensor(rho, dtype=torch.float32).to(device)

    r0 = 5.0
    n_frames = 12

    imgs = []
    psfs = []
    otfs = []

    for i in range(n_frames):
        
        # Generate random PSF
        wavefront, psf = conv.compute_random_psf_r0(r0=r0, 
                                                    remove_tt=True, 
                                                    diffraction=False)

        # Convolve the image with the PSF
        out, otf = conv.convolve(img, psf)

        out = out.cpu().numpy()
        
        out += noise * np.random.standard_normal(out.shape)

        imgs.append(out[None, :, :])
        psfs.append(psf[None, :, :].cpu().numpy())
        otfs.append(otf[None, :, :].cpu().numpy())

    imgs = np.concatenate(imgs, axis=0)
    psfs = np.concatenate(psfs, axis=0)
    otfs = np.concatenate(otfs, axis=0)

    imgs = torch.tensor(imgs, dtype=torch.float32).to(device)
    otfs = torch.tensor(otfs, dtype=torch.complex64).to(device)
        
    imgs_ft = torch.fft.fft2(imgs, norm='ortho')
    
    Ks = np.logspace(-1, 5, 100)
    v0s = np.logspace(-3, 3, 100)

    KK, v0v0 = np.meshgrid(Ks, v0s, indexing='ij')
    
    loss = torch.zeros((len(Ks), len(v0s))).to(device)
    loss_data = torch.zeros_like(loss)
    loss_marginal = torch.zeros_like(loss)

    for j in range(len(v0s)):        
        for i in range(len(Ks)):
            v0 = v0v0[i, j]
            K = KK[i, j]

            s_u = K / (1.0 + (rho/v0)**2)**(beta_orig / 2.0)

            du = imgs_ft        
            hu2 = noise**2 + s_u * torch.sum(otfs * torch.conj(otfs), dim=0)
            du2 = torch.sum(du * torch.conj(du), dim=0)
            hu_du = torch.sum(du * torch.conj(otfs), dim=0)
            hu_du2 = s_u * hu_du * torch.conj(hu_du)
            
            tmp_data = 0.5 * (du2 - hu_du2 / hu2) / noise**2

            tmp_marginal = 0.5 * torch.log(hu2)

                                    
            loss_data[i, j] = torch.mean(tmp_data[1:, 1:].real)
            loss_marginal[i, j] = torch.mean(tmp_marginal[1:, 1:].real)
            loss[i, j] = loss_data[i, j] + loss_marginal[i, j]
            
        
    loss = loss.cpu().numpy()
    loss_data = loss_data.cpu().numpy()
    loss_marginal = loss_marginal.cpu().numpy()
    imgs = imgs.cpu().numpy()

    min_idx = np.unravel_index(np.argmin(loss), loss.shape)
    K_est = KK[min_idx[0], min_idx[1]]
    v0_est = v0v0[min_idx[0], min_idx[1]]
    print(np.min(loss), K_est, v0_est, loss[min_idx])
    
    # loss -= np.min(loss)    
    
    fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(15, 15))
    ax[0, 0].imshow(img.cpu().numpy(), cmap='gray')
    ax[0, 0].set_title('Original image')
    ax[0, 1].imshow(imgs[0], cmap='gray')
    ax[0, 1].set_title('Convolved image 1')
    ax[0, 2].imshow(imgs[1], cmap='gray')
    ax[0, 2].set_title('Convolved image 2')

    im = ax[1, 0].imshow(loss_data, origin='lower', extent=(np.log10(v0s[0]), np.log10(v0s[-1]), np.log10(Ks[0]), np.log10(Ks[-1])))
    ax[1, 0].set_xlabel('v0')
    ax[1, 0].set_ylabel('K')
    ax[1, 0].set_title('Data term')
    ax[1, 0].plot(np.log10(v0_est), np.log10(K_est), 'go', label='Estimated parameters')
    ax[1, 0].plot(np.log10(v0_orig), np.log10(K_orig), 'rx', label='True parameters')
    

    im = ax[1, 1].imshow(loss_marginal, origin='lower', extent=(np.log10(v0s[0]), np.log10(v0s[-1]), np.log10(Ks[0]), np.log10(Ks[-1])))
    ax[1, 1].set_xlabel('v0')
    ax[1, 1].set_ylabel('K')
    ax[1, 1].set_title('Marginal term')
    ax[1, 1].plot(np.log10(v0_est), np.log10(K_est), 'go', label='Estimated parameters')
    ax[1, 1].plot(np.log10(v0_orig), np.log10(K_orig), 'rx', label='True parameters')    
    
    im = ax[1, 2].imshow(loss, origin='lower', extent=(np.log10(v0s[0]), np.log10(v0s[-1]), np.log10(Ks[0]), np.log10(Ks[-1])))    
    levels = np.arange(20) * 0.05 + np.min(loss)
    ax[1, 2].contour(loss, origin='lower', extent=(np.log10(v0s[0]), np.log10(v0s[-1]), np.log10(Ks[0]), np.log10(Ks[-1])), colors='white', alpha=0.3, levels=levels)
    ax[1, 2].set_xlabel('v0')
    ax[1, 2].set_ylabel('K')
    ax[1, 2].set_title('Combined term')    
    ax[1, 2].plot(np.log10(v0_est), np.log10(K_est), 'go', label='Estimated parameters')
    ax[1, 2].plot(np.log10(v0_orig), np.log10(K_orig), 'rx', label='True parameters')

