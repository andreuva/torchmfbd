import numpy as np
import util
import kl_modes
import matplotlib.pyplot as pl

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

        # Generate Hamming window function for WFS correlation
        print("Generating apodization window...")
        self.npix_apod = self.config['npix_apodization']
        win = np.hanning(self.npix_apod)
        winOut = np.ones(self.config['n_pixel'])
        winOut[0:self.npix_apod//2] = win[0:self.npix_apod//2]
        winOut[-self.npix_apod//2:] = win[-self.npix_apod//2:]
        self.window = np.outer(winOut, winOut)

        # Compute the overfill to properly generate the PSFs from the wavefronts
        print("Computing telescope apertures...")
        self.n_wavelength = len(self.config['wavelength'])

        self.pupil = []
        self.basis = []
        self.mask = []

        for i in range(self.n_wavelength):
            print(f"Wavelength: {self.config['wavelength'][i]} - D: {self.config['diameter'][i]} - pix: {self.config['pix_size'][i]}")
            overfill = util.psf_scale(self.config['wavelength'][i], 
                                            self.config['diameter'][i], 
                                            self.config['pix_size'][i])
            
            if (overfill < 1.0):
                raise Exception(f"The pixel size is not small enough to model a telescope with D={self.telescope_diameter} cm")

            # Compute telescope aperture
            pupil = util.aperture(npix=self.config['n_pixel'], 
                            cent_obs = self.config['central_obs'][i] / self.config['diameter'][i], 
                            spider=0, 
                            overfill=overfill)
            self.pupil.append(pupil)
            
            print(f"Computing KL modes...")
            self.kl = kl_modes.KL()
            basis = self.kl.precalculate(npix_image = self.config['n_pixel'], 
                                n_modes_max = self.config['n_modes'], 
                                first_noll = 2, 
                                overfill=overfill)
            basis /= np.max(np.abs(basis), axis=(1, 2), keepdims=True)
            
            self.basis.append(basis)

            cutoff = self.config['diameter'][i] / (self.config['wavelength'][i] * 1e-8) / 206265.0
            freq = np.fft.fftfreq(self.config['n_pixel'], d=self.config['pix_size'][i]) / cutoff
                
            xx, yy = np.meshgrid(freq, freq)
            rho = np.sqrt(xx ** 2 + yy ** 2)
            self.mask.append(rho <= self.config['mask_cutoff'])

        self.r0_min = self.config['r0_min']
        self.r0_max = self.config['r0_max']

        self.npix_out = self.config['n_pixel_out']        

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
        wavefront = np.einsum('i,ilm->lm', modes, basis)

        # Compute the complex phase
        phase = pupil * np.exp(1j * wavefront)

        # Compute FFT of the pupil function and compute autocorrelation
        ft = np.fft.fft2(phase)
        psf = (np.conj(ft) * ft).real
        
        # Normalize PSF to unit amplitude        
        psf_norm = psf / np.sum(psf)
        
        return wavefront, psf_norm
    
    def gen_psf(self, r0, wl=2):

        coef = (self.config['diameter'][wl] / r0)**(5.0/6.0)

        sigma_KL = coef * np.sqrt(self.kl.varKL)


        modes = np.random.normal(loc=0.0, scale=sigma_KL, size=sigma_KL.shape)
                        
        # Compute PSF and convolve with original image            
        wavefront, psf = self.compute_psfs(modes, self.pupil[wl], self.basis[wl])
        
        return psf