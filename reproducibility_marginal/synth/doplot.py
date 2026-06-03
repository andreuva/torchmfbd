import numpy as np
import matplotlib.pyplot as pl
import h5py
from astropy.io import fits
from scipy import ndimage
import torchmfbd


def plot_r0():
    image = np.array(fits.open('FeI_6302_cont.fits')[0].data).astype(np.float32)
    pix_simulation = 0.0139
    pix_crisp = 0.059
    zoom = pix_simulation / pix_crisp
    image = ndimage.zoom(image, zoom=zoom, order=1)
    diff_crisp = 1.22 * 6302e-8 / 100.0 * 206265.0

    r0s = [10, 15, 20, 30]
    noise = [0.001, 0.01, 0.1]

    labels = ['Frame'] + [fr"$\sigma$={n}" for n in noise]

    vmin = np.min(image)
    vmax = np.max(image)

    # Read all files
    f_marginal = []
    f_joint = []

    for r0 in r0s:
        f_marginal.append(h5py.File(f'marginal_r0_{r0}.h5', 'r'))

    for r0 in r0s:
        f_joint.append(h5py.File(f'joint_r0_{r0}.h5', 'r'))

    # MARGINAL PLOTS
    fig, ax = pl.subplots(nrows=4, ncols=5, figsize=(22, 13))
    for i, r0 in enumerate(r0s):
        ax[0, i].imshow(f_marginal[i]['best_frame'][0, ...], vmin=vmin, vmax=vmax)
        for j, n in enumerate(noise):
            ax[j+1, i].imshow(f_marginal[i]['obj'][j, ...], vmin=vmin, vmax=vmax)

        ax[0, i].set_title(f"r0 = {r0} cm")


    for j in range(4):
        ax[j, -1].imshow(image, vmin=vmin, vmax=vmax)
            
        ax[j, 0].text(0.1, 0.95, labels[j],
                            transform=ax[j, 0].transAxes, 
                            fontsize=18, 
                            verticalalignment='top', 
                            color='yellow',
                            fontweight='bold')
    ax[0, -1].set_title("Original Image")

    # Adjust layout first to fix panel sizes
    fig.subplots_adjust(left=0.05, right=0.88, wspace=0.1, hspace=0.15)

    for j in range(4):
        # Create an axes for the colorbar to the right of the last subplot in each row
        pos = ax[j, -1].get_position()
        # Define cax relative to the adjusted panel position
        cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.015, pos.height])
        
        # Use the last image in the row to define the colorbar scale
        im = ax[j, -1].get_images()[0]
        fig.colorbar(im, cax=cax)

    fig.suptitle("Marginal Deconvolution", fontsize=20)

    # MARGINAL PLOTS
    fig, ax = pl.subplots(nrows=4, ncols=5, figsize=(22, 13))
    for i, r0 in enumerate(r0s):
        ax[0, i].imshow(f_joint[i]['best_frame'][0, ...], vmin=vmin, vmax=vmax)
        for j, n in enumerate(noise):
            ax[j+1, i].imshow(f_joint[i]['obj'][j, ...], vmin=vmin, vmax=vmax)

        ax[0, i].set_title(f"r0 = {r0} cm")


    for j in range(4):
        ax[j, -1].imshow(image, vmin=vmin, vmax=vmax)
            
        ax[j, 0].text(0.1, 0.95, labels[j],
                            transform=ax[j, 0].transAxes, 
                            fontsize=18, 
                            verticalalignment='top', 
                            color='yellow',
                            fontweight='bold')
    ax[0, -1].set_title("Original Image")

    # Adjust layout first to fix panel sizes
    fig.subplots_adjust(left=0.05, right=0.88, wspace=0.1, hspace=0.15)

    for j in range(4):
        # Create an axes for the colorbar to the right of the last subplot in each row
        pos = ax[j, -1].get_position()
        # Define cax relative to the adjusted panel position
        cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.015, pos.height])
        
        # Use the last image in the row to define the colorbar scale
        im = ax[j, -1].get_images()[0]
        fig.colorbar(im, cax=cax)

    fig.suptitle("Joint Deconvolution", fontsize=20)


    # POWER SPECTRA
    fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(15, 8), tight_layout=True, sharex=True, sharey=True)

    for i, n in enumerate(noise):
        im = image
        kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)    
        ax[0, i].loglog(kk, power / 10.0**np.nanmean(np.log10(power[5:8])) , label='Original', linewidth=2, color='C0')
            
        for j in range(4):

            im = f_marginal[j]['best_frame'][i, ...]
            kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)    
            ax[0, i].loglog(kk, power / 10.0**np.nanmean(np.log10(power[5:8])), '--', linewidth=2, color=f'C{j+1}')
            
            im = f_marginal[j]['obj'][i, ...]
            kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)    
            ax[0, i].loglog(kk, power / 10.0**np.nanmean(np.log10(power[5:8])), label=fr'r0={r0s[j]} cm', linewidth=2, color=f'C{j+1}')

        ax[0, i].axvline(1.0 / (diff_crisp / pix_crisp), color='black')

        ax[0, i].set_title(fr"$\sigma$={n}")

        ax[0, 0].text(0.03, 0.1, 'Marginal',
                            transform=ax[0, 0].transAxes, 
                            fontsize=14, 
                            verticalalignment='top', 
                            color='black',
                            fontweight='bold')

    for i, n in enumerate(noise):
        im = image
        kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)    
        ax[1, i].loglog(kk, power / 10.0**np.nanmean(np.log10(power[5:8])) , label='Original', linewidth=2, color='C0')
            
        for j in range(4):

            im = f_joint[j]['best_frame'][i, ...]
            kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)    
            ax[1, i].loglog(kk, power / 10.0**np.nanmean(np.log10(power[5:8])), '--', linewidth=2, color=f'C{j+1}')
            
            im = f_joint[j]['obj'][i, ...]
            kk, power = torchmfbd.util.azimuthal_power(im / np.nanmean(im), apodization=10, angles=[-45,45], range_angles=15)    
            ax[1, i].loglog(kk, power / 10.0**np.nanmean(np.log10(power[5:8])), label=fr'r0={r0s[j]} cm', linewidth=2, color=f'C{j+1}')

        ax[1, i].axvline(1.0 / (diff_crisp / pix_crisp), color='black')

        ax[1, 0].text(0.03, 0.1, 'Joint',
                            transform=ax[1, 0].transAxes, 
                            fontsize=14, 
                            verticalalignment='top', 
                            color='black',
                            fontweight='bold')

        
    ax[0 ,0].set_xlim([1e-2, 0.6])
    ax[0, 1].legend()

    # Close files
    for file in f_joint:
        file.close()
    for file in f_marginal:
        file.close()

def plot_K():
    image = np.array(fits.open('FeI_6302_cont.fits')[0].data).astype(np.float32)
    pix_simulation = 0.0139
    pix_crisp = 0.059
    zoom = pix_simulation / pix_crisp
    image = ndimage.zoom(image, zoom=zoom, order=1)

    r0s = [10, 15, 20, 30]

    f_marginal = []
    f_joint = []

    for r0 in r0s:

        f = h5py.File(f'nmodes_marginal_r0_{r0}.h5', 'r')
        f_marginal.append(f)
        f = h5py.File(f'nmodes_joint_r0_{r0}.h5', 'r')
        f_joint.append(f)

    modes = [5, 9, 14, 20, 27, 35, 44, 65, 90, 119]
    
    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(12, 12), tight_layout=True)
    
    for i in range(len(r0s)):
        ax[0, 0].plot(modes, f_marginal[i]['loss'][:], label=fr'r0={r0s[i]} cm', linewidth=2)
        ax[0, 1].plot(modes, f_joint[i]['loss'][:, -1], label=fr'r0={r0s[i]} cm', linewidth=2)

        std_joint = np.zeros(len(modes))
        std_marginal = np.zeros(len(modes))

        for j in range(len(modes)):
            res_joint = f_joint[i]['obj'][j, 5:-5, 5:-5] - image[5:-5, 5:-5]
            res_marginal = f_marginal[i]['obj'][j, 5:-5, 5:-5] - image[5:-5, 5:-5]

            std_joint[j] = np.std(res_joint)
            std_marginal[j] = np.std(res_marginal)

        ax[1, 0].plot(modes, std_marginal, label=fr'r$_0$={r0s[i]} cm', linewidth=2)
        ax[1, 1].plot(modes, std_joint, label=fr'r$_0$={r0s[i]} cm', linewidth=2)
    
    ax[0, 0].set_title('Marginal')
    
    ax[0, 0].set_ylabel('Final Loss')
    ax[0, 1].set_title('Joint')

    ax[1, 0].set_ylim([0, 0.2])
    ax[1, 1].set_ylim([0, 0.2])
    ax[1, 0].legend()

    ax[1, 0].set_ylabel('Std of Residuals')
    
    fig.supxlabel('Number of KL modes')

    pl.savefig('nmodes.pdf', dpi=300)

    # fig, ax = pl.subplots(nrows=4, ncols=len(modes)+1, figsize=(24, 16), tight_layout=True, sharex=True, sharey=True)
    # for i, n_m in enumerate(modes):
    #     ax[0, i].imshow(f_marginal['obj'][i, 5:-5, 5:-5], vmin=0.7, vmax=1.7)
    #     ax[0, i].set_title(f"{n_m} modes")
    #     ax[2, i].imshow(f_joint['obj'][i, 5:-5, 5:-5], vmin=0.7, vmax=1.7)
        
    #     ax[1, i].imshow(f_marginal['obj'][i, 5:-5, 5:-5] - image[5:-5, 5:-5], vmin=-0.1, vmax=0.1)        
    #     ax[1, i].set_title(f"<{np.std(f_marginal['obj'][i, 5:-5, 5:-5] - image[5:-5, 5:-5]):.3f}>")
    #     ax[3, i].imshow(f_joint['obj'][i, 5:-5, 5:-5] - image[5:-5, 5:-5], vmin=-0.1, vmax=0.1)
    #     ax[3, i].set_title(f"<{np.std(f_joint['obj'][i, 5:-5, 5:-5] - image[5:-5, 5:-5]):.3f}>")

    # for i in range(4):
    #     ax[i, -1].imshow(image[5:-5, 5:-5], vmin=0.7, vmax=1.7)
    # ax[0, -1].set_title("Original Image")

    # ax[0, 0].set_ylabel('Marginal')
    # ax[2, 0].set_ylabel('Joint')
    
    for file in f_marginal:
        file.close()
    for file in f_joint:
        file.close()
    
if __name__ == '__main__':
    pl.close('all')
    plot_r0()
    # plot_K()
    