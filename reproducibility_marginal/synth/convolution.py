import numpy as np
import matplotlib.pyplot as pl
from tqdm import tqdm
import util
from astropy.io import fits
import torch
import zern
import os
import scipy.special as sp
import scipy.stats as stats
from scipy import ndimage


class Convolution(object):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self, config):
        """
        Very simple training set made of 200 Gaussians of width between 0.5 and 1.5
        We later augment this with a velocity and amplitude.
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super(Convolution, self).__init__()

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Compute the overfill to properly generate the PSFs from the wavefronts
        print("Computing telescope aperture...")
        
        print(f"Wavelength: {self.config['wavelength']} A - D: {self.config['diameter']} - pix: {self.config['pix_size']}")
        self.diff_limit = 1.22 * self.config['wavelength'] * 1e-8 / self.config['diameter'] * 206265
        
        print(f"Diff. limit : {self.diff_limit}")
        
        overfill = util.psf_scale(self.config['wavelength'], 
                                        self.config['diameter'], 
                                        self.config['pix_size'])

        print(f"Overfill : {overfill}")
                
        if (overfill < 1.0):
            raise Exception(f"The pixel size is not small enough to model a telescope with D={self.telescope_diameter} cm")

        # Compute telescope aperture
        pupil = util.aperture(npix=self.config['n_pixel'], 
                        cent_obs = self.config['central_obs'] / self.config['diameter'], 
                        spider=0,
                        overfill=overfill)
        self.pupil = torch.tensor(pupil, dtype=torch.float32).to(self.device)

        self.npix = self.config['n_pixel']
        self.n_modes = self.config['n_modes']
        self.telescope_diameter = self.config['diameter']

        if (self.config['basis'] == 'zernike'):

            print("Computing covariance matrix...")
            self.init_covariance()
                                
            if os.path.exists(f"zernike_basis_{int(self.config['diameter'])}_{int(self.npix)}.npz"):
                print("Loading precomputed Zernike basis...")
                tmp = np.load(f"zernike_basis_{int(self.config['diameter'])}_{int(self.npix)}.npz")
                basis = tmp['basis']            
            else:
                print(f"Computing Zernike modes...")
                basis = self.precalculate_zernike(overfill=overfill)
                np.savez(f"zernike_basis_{int(self.config['diameter'])}_{int(self.npix)}.npz", basis=basis, variance=np.diag(self.covariance))

        if (self.config['basis'] == 'kl'):
            
            if os.path.exists(f"kl_basis_{int(self.config['diameter'])}_{int(self.npix)}.npz"):
                print("Loading precomputed KL basis...")
                tmp = np.load(f"kl_basis_{int(self.config['diameter'])}_{int(self.npix)}.npz")
                basis = tmp['basis']
                self.varKL = tmp['variance']
            else:
                print(f"Computing KL modes...")
                self.kl = kl_modes.KL()
                basis = self.kl.precalculate(npix_image = self.config['n_pixel'], 
                                n_modes_max = self.config['n_modes'],                                
                                overfill=overfill)
                self.varKL = self.kl.varKL                
                np.savez(f"kl_basis_{int(self.config['diameter'])}_{int(self.npix)}.npz", basis=basis, variance=self.kl.varKL)

        self.basis = torch.tensor(basis, dtype=torch.float32).to(self.device)        

        # Apodization window to reduce edge effects in FFTs
        self.npix_apod = self.config['apodization']
        win = np.hanning(2*self.npix_apod)
        winOut = np.ones(self.npix)
        winOut[0:self.npix_apod] = win[0:self.npix_apod]
        winOut[-self.npix_apod:] = win[-self.npix_apod:]
        self.window = np.outer(winOut, winOut)
        self.window = torch.tensor(self.window, dtype=torch.float32, device=self.device)
                        
    def precalculate_zernike(self, overfill):
        
        Z_machine = zern.ZernikeNaive(mask=[])
        x = np.linspace(-1, 1, self.npix)
        xx, yy = np.meshgrid(x, x)
        rho = overfill * np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        aperture_mask = rho <= 1.0

        Z = np.zeros((self.n_modes, self.npix, self.npix))

        # Do not take into account the piston
        noll_Z = 2 + np.arange(self.n_modes)

        for mode in tqdm(range(self.n_modes)):
                                                
            jz = noll_Z[mode]
            n, m = zern.zernIndex(jz)
            Zmode = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
            Z[mode, :, :] = Zmode * aperture_mask
                
        return Z

    def _even(self, x):
        return x%2 == 0

    def _zernike_parity(self, j, jp):
        return self._even(j-jp)
    
    def init_covariance(self):
        """
        Fill the covariance matrix for Kolmogorov turbulence
        Args:
            r0 (float): Fried parameter (cm)
        Returns:
            N/A
        """
        self.covariance = np.zeros((self.n_modes + 1,self.n_modes + 1))

        for j in range(self.n_modes + 1):
            n, m = zern.zernIndex(j + 1)
            for jpr in range(self.n_modes + 1):
                npr, mpr = zern.zernIndex(jpr + 1)
                
                deltaz = (m == mpr) and (self._zernike_parity(j, jpr) or m == 0)
                
                if (deltaz):                
                    phase = (-1.0)**(0.5*(n+npr-2*m))
                    t1 = np.sqrt((n+1)*(npr+1)) 
                    t2 = sp.gamma(14./3.0) * sp.gamma(11./6.0)**2 * (24.0/5.0*sp.gamma(6.0/5.0))**(5.0/6.0) / (2.0*np.pi**2)

                    Kzz = t2 * t1 * phase
                    
                    t1 = sp.gamma(0.5*(n+npr-5.0/3.0))
                    t2 = sp.gamma(0.5*(n-npr+17.0/3.0)) * sp.gamma(0.5*(npr-n+17.0/3.0)) * sp.gamma(0.5*(n+npr+23.0/3.0))
                    self.covariance[j,jpr] = Kzz * t1 / t2

        self.covariance[0, 0] = 1.0
        self.covariance[0, :] = 0.0
        self.covariance[:, 0] = 0.0

        # Remove piston
        self.covariance = self.covariance[1:, 1:]

        self.random_zernike = stats.multivariate_normal(mean=np.zeros(self.n_modes), cov=self.covariance, allow_singular=True)

    def compute_psfs(self, modes, pupil, basis):
        """Compute the PSFs and their Fourier transform from a set of modes
        
        Args:
            wavefront_focused ([type]): wavefront of the focused image
            illum ([type]): pupil aperture
            diversity ([type]): diversity for this specific images
        
        """

        # --------------
        # Focused PSF
        # --------------
        # Compute wavefronts from estimated modes                
        wavefront = torch.einsum('i,ilm->lm', modes, basis)

        # Compute the complex phase
        phase = pupil * torch.exp(1j * wavefront)

        # Compute FFT of the pupil function and compute autocorrelation
        ft = torch.fft.fft2(phase)
        psf = (torch.conj(ft) * ft).real
        
        # Normalize PSF to unit amplitude        
        psf_norm = psf / torch.sum(psf)
        
        return wavefront, psf_norm
    
    def compute_random_psf_r0(self, r0, remove_tt=False, diffraction=False):
        """Compute the PSFs and their Fourier transform from a set of modes
        
        Args:
            wavefront_focused ([type]): wavefront of the focused image
            illum ([type]): pupil aperture
            diversity ([type]): diversity for this specific images
        
        """

        # (D/r0)**(5.0/3.0) is the variance of the Zernike modes
        # For this reason, we multiply by sqrt((D/r0)**(5.0/3.0)) to get the actual modes
        modes = (self.telescope_diameter / r0)**(5.0/6.0) * self.random_zernike.rvs()

        if (diffraction):
            modes *= 0.0

        modes_th = torch.tensor(modes, dtype=torch.float32).to(self.device)

        if (remove_tt):
            modes_th[0:2] = 0.0

        # Compute PSF and convolve with original image
        wavefront, psf = self.compute_psfs(modes_th, self.pupil, self.basis)
        
        return wavefront, psf
            
    def convolve(self, image, psf):

        # Move tensors to GPU if available
        image = torch.tensor(image, dtype=torch.float32).to(self.device)

        # Compute mean value of the image and subtract it to reduce edge effects in the FFTs
        mean_val = torch.mean(image, dim=(-1, -2), keepdim=True)
        tmp = image - mean_val
        apod = tmp * self.window
        apod += mean_val
        
        fft_image = torch.fft.fft2(apod)
        
        convolved = torch.fft.ifft2(torch.fft.fft2(psf) * fft_image).real.cpu().numpy()
        
        return convolved
                            

if (__name__ == '__main__'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration file
    config = {
            'n_pixel': 1152,
            'wavelength': 6302,
            'apodization': 12,
            'diameter': 100.0,
            'pix_size': 0.0139,
            'central_obs': 0.0,
            'basis': 'zernike',
            'n_modes': np.sum(np.arange(20)+2),
        }
    
    image = np.array(fits.open('FeI_6302_cont.fits')[0].data).astype(np.float32)
    
    conv = Convolution(config)

    r0s = np.array([10.0, 15.0, 20.0, 30.0])
    n_frames = 10

    for r0 in tqdm(r0s):

        out_all = []

        for i in range(n_frames):
    
            # Generate random PSF
            wavefront, psf = conv.compute_random_psf_r0(r0=r0, 
                                                        remove_tt=True, 
                                                        diffraction=False)

            # Convolve the image with the PSF
            out = conv.convolve(image, psf)
            
            # REBIN IMAGE TO SST PIXEL SIZE
            zoom = 0.0139 / 0.059

            out = ndimage.zoom(out, zoom=zoom, order=1)

            out_all.append(out[None, :, :])

        np.savez(f"convolved_r0_{int(r0)}.npz", convolved=np.concatenate(out_all, axis=0))
    
    # fig, ax = pl.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    # ax[0].imshow(image, cmap='gray')
    # ax[0].set_title("Original image")

    # ax[1].imshow(out, cmap='gray')
    # ax[1].set_title("Convolved image")

    
