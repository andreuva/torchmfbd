import torch
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec


def export_psfs(decSI):
    """
    Concatenates all PSF batches from decSI.psf_seq into a single array per object,
    saves the result to a compressed .npz file, and returns the arrays.

    Returns
    -------
    psfs : list of np.ndarray
        One array per object, shape (n_patches, n_frames, nx, ny), peak-normalised.
    """
    model = decSI.psf_model.upper()

    psfs = []
    for i in range(decSI.n_o):
        obj_psf = torch.cat([batch[i] for batch in decSI.psf_seq], dim=0).cpu().numpy()
        
        # Check if all patches are identical (potential bug indicator)
        if obj_psf.shape[0] > 1:
            diff = np.abs(obj_psf[1:] - obj_psf[0:1]).max()
            if diff < 1e-10:
                print(f'[WARNING] Object {i}: All {obj_psf.shape[0]} PSFs are numerically IDENTICAL! (max diff: {diff})')
            else:
                print(f'[PSF] Object {i}: PSFs are unique (max patch diff: {diff:.2e})')

        # Keep the original area=1 normalization to allow physical peak comparisons
        psfs.append(obj_psf)

    fname = f'psf_data_{model}.npz'
    np.savez_compressed(fname, psfs=np.stack(psfs))
    print(f'[PSF] Saved {fname}  shape: {psfs[0].shape}')

    return psfs


def plot_psfs(psfs, model_name, n_patches=6, frame_idx=0, filename=None):
    """
    Plots a grid of PSFs: rows = objects, columns = sampled patches.

    Parameters
    ----------
    psfs : list of np.ndarray
        Output of export_psfs().
    model_name : str
        Label for the plot title (e.g. 'KL' or 'NMF').
    n_patches : int
        Number of evenly spaced patches to display.
    frame_idx : int
        Which frame to display from each patch.
    filename : str or None
        If provided, saves the figure to this path.
    """
    n_obj = len(psfs)
    total_patches = psfs[0].shape[0]
    patch_indices = np.linspace(0, total_patches - 1, n_patches, dtype=int)

    # Find global vmax across all displayed patches for a consistent colorbar
    # This allows comparing the "peakiness" of different PSFs
    vmax = 0
    for i in range(n_obj):
        vmax = max(vmax, psfs[i][patch_indices, frame_idx].max())

    fig = pl.figure(figsize=(3 * n_patches, 3 * n_obj))
    gs = gridspec.GridSpec(n_obj, n_patches, hspace=0.05, wspace=0.05)

    for row, i in enumerate(range(n_obj)):
        for col, p_idx in enumerate(patch_indices):
            ax = fig.add_subplot(gs[row, col])
            img = psfs[i][p_idx, frame_idx]
            # Shift from corner to center for visualization
            img = np.fft.fftshift(img)
            nx, ny = img.shape
            
            ax.imshow(img , cmap='nipy_spectral', interpolation='nearest', origin='lower', vmax=vmax)
            
            # Add centering crosshairs to check for asymmetries
            ax.axvline(ny // 2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.axhline(nx // 2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f'P{p_idx}', fontsize=8)
            if col == 0:
                ax.set_ylabel(f'Obj {i}', fontsize=8)

    fig.suptitle(f'{model_name} PSFs  (frame {frame_idx})', fontsize=11, y=1.01)

    fname = filename or f'psf_plot_{model_name}.png'
    pl.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'[PSF] Plot saved to {fname}')
    pl.close(fig)
