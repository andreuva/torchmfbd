import os
import argparse
import numpy as np
import torch
from astropy.io import fits
from tqdm import tqdm
import torchmfbd

def read_hifi_dataset(fits_path, n_frames=100, crop_size=None):
    """
    Reads HiFI+ level1 dataset from a .fts file.
    Camera 1 (Ext 1, 3, 5...) -> Narrow-band (656.3 nm Lyot)
    Camera 2 (Ext 2, 4, 6...) -> Broad-band (656.7 nm Wideband)
    """
    f = fits.open(fits_path)
    total_ext = len(f) - 1  # total ImageHDUs
    max_frames = min(total_ext // 2, n_frames)
    
    img_h, img_w = f[1].data.shape
    if crop_size is not None:
        ch, cw = min(img_h, crop_size), min(img_w, crop_size)
    else:
        ch, cw = img_h, img_w

    nb_frames = np.zeros((1, max_frames, ch, cw), dtype=np.float32)
    wb_frames = np.zeros((1, max_frames, ch, cw), dtype=np.float32)

    for i in tqdm(range(max_frames), desc="Loading FITS extensions"):
        # Ext 1 + 2*i: Camera 1 (Narrowband)
        # Ext 2 + 2*i: Camera 2 (Broadband)
        nb_frames[0, i, :, :] = f[1 + 2*i].data[:ch, :cw]
        wb_frames[0, i, :, :] = f[2 + 2*i].data[:ch, :cw]

    header = f[0].header
    f.close()
    return nb_frames, wb_frames, header

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MOMFBD on HiFI+ Level 1 dataset (with off-limb enhancements).")
    parser.add_argument("--fits", type=str, 
                        default="/dat/andreuva/data/hifiplus/level1/20260714/hifiplus2_20260714_080658_sd.fts",
                        help="Path to HiFI+ FITS dataset file")
    parser.add_argument("--config", type=str, default="hifi_momfbd.yaml", help="Path to config YAML file")
    parser.add_argument("--n_frames", type=int, default=100, help="Number of frames per camera (e.g. 80-100 for poor seeing)")
    parser.add_argument("--patch_size", type=int, default=96, help="Patch size for sub-field deconvolution")
    parser.add_argument("--stride_size", type=int, default=50, help="Stride size for overlapping patches")
    parser.add_argument("--crop_size", type=int, default=None, help="Crop region size (e.g., 512 for fast CPU testing)")
    parser.add_argument("--no_destretch", action="store_true", help="Disable destretching")
    parser.add_argument("--off_limb", action="store_true", default=True, help="Enable off-limb optimizations (percentile scaling, Lyot ranking, smooth flow)")
    parser.add_argument("--n_iterations", type=int, default=250, help="Number of optimization iterations")
    parser.add_argument("--simultaneous_seq", type=int, default=200, help="Simultaneous patch sequences to process")
    parser.add_argument("--output", type=str, default="hifi_momfbd_result.fits", help="Output FITS path")
    args = parser.parse_args()

    print(f"Reading dataset from {args.fits}...")
    nb_frames, wb_frames, header = read_hifi_dataset(args.fits, n_frames=args.n_frames, crop_size=args.crop_size)

    # Frame quality ranking
    if args.off_limb:
        print("Off-limb mode enabled: Ranking frames by Narrow-band Lyot spatial variance...")
        # Off-limb: use spatial variance of Narrow-band (Lyot 656.3 nm) where bright spicules/features appear
        quality_score = np.var(nb_frames[0, ...], axis=(-1, -2))
    else:
        # On-disk: use standard broad-band contrast std/mean
        quality_score = np.std(wb_frames[0, ...], axis=(-1, -2)) / np.maximum(np.mean(wb_frames[0, ...], axis=(-1, -2)), 1e-5)

    sort_ind = np.argsort(quality_score)[::-1]

    wb_frames = wb_frames[:, sort_ind, :, :]
    nb_frames = nb_frames[:, sort_ind, :, :]

    # Convert to PyTorch tensors
    wb_tensor = torch.tensor(wb_frames, dtype=torch.float32)
    nb_tensor = torch.tensor(nb_frames, dtype=torch.float32)

    # Intensity scaling / normalization
    if args.off_limb:
        # Scale by 95th percentile intensity to prevent dark sky zeros from corrupting frame mean
        p95_wb = np.percentile(wb_frames, 95, axis=(-2, -1), keepdims=True)
        p95_nb = np.percentile(nb_frames, 95, axis=(-2, -1), keepdims=True)
        wb_tensor /= torch.tensor(np.maximum(p95_wb, 1e-5), dtype=torch.float32)
        nb_tensor /= torch.tensor(np.maximum(p95_nb, 1e-5), dtype=torch.float32)
    else:
        wb_tensor /= torch.mean(wb_tensor, dim=(-2, -1), keepdim=True)
        nb_tensor /= torch.mean(nb_tensor, dim=(-2, -1), keepdim=True)

    # Destretch / alignment if enabled
    if not args.no_destretch:
        lambda_tt = 0.08 if args.off_limb else 0.01
        print(f"Applying destretching across time series (lambda_tt={lambda_tt})...")
        # Destretch WB sequence
        wb_warped, _ = torchmfbd.destretch(
            wb_tensor[:, None, :, :, :],
            ngrid=64, lr=0.50, reference_frame=0, border=6, n_iterations=40, lambda_tt=lambda_tt
        )
        wb_tensor = wb_warped[:, 0, :, :, :]

        # Destretch NB sequence
        nb_warped, _ = torchmfbd.destretch(
            nb_tensor[:, None, :, :, :],
            ngrid=64, lr=0.50, reference_frame=0, border=6, n_iterations=40, lambda_tt=lambda_tt
        )
        nb_tensor = nb_warped[:, 0, :, :, :]

    # Create deconvolution object
    decSI = torchmfbd.Deconvolution(args.config)

    # Patchify
    patchify = torchmfbd.Patchify4D()
    wb_patches = patchify.patchify(wb_tensor, patch_size=args.patch_size, stride_size=args.stride_size, flatten_sequences=True)
    nb_patches = patchify.patchify(nb_tensor, patch_size=args.patch_size, stride_size=args.stride_size, flatten_sequences=True)

    # Estimate noise
    wb_noise = torchmfbd.compute_noise(wb_patches[0:1, 0:1, ...])
    nb_noise = torchmfbd.compute_noise(nb_patches[0:1, 0:1, ...])

    # Add frames: Object 0 = Wideband, Object 1 = Narrowband
    print("Adding frames to MOMFBD engine...")
    decSI.add_frames(wb_patches, id_object=0, id_diversity=0, diversity=0.0, sigma=wb_noise)
    decSI.add_frames(nb_patches, id_object=1, id_diversity=0, diversity=0.0, sigma=nb_noise)

    # Deconvolve
    print("Starting MOMFBD deconvolution...")
    decSI.deconvolve(
        infer_object=False,
        optimizer='adam',
        simultaneous_sequences=args.simultaneous_seq,
        n_iterations=args.n_iterations
    )

    # Unpatchify results
    print("Reconstructing deconvolved objects...")
    obj_wb = patchify.unpatchify(decSI.obj[0][:, None, ...], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy()
    obj_nb = patchify.unpatchify(decSI.obj[1][:, None, ...], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy()

    # Save reconstructed objects to FITS
    hdu0 = fits.PrimaryHDU(header=header)
    hdu1 = fits.ImageHDU(data=obj_wb[0, :, :], name="WIDEBAND_RECONSTRUCTED")
    hdu2 = fits.ImageHDU(data=obj_nb[0, :, :], name="NARROWBAND_RECONSTRUCTED")
    hdu3 = fits.ImageHDU(data=decSI.rho[0].cpu().numpy(), name="PHASE_MODES_WB")
    hdu4 = fits.ImageHDU(data=decSI.rho[1].cpu().numpy(), name="PHASE_MODES_NB")
    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4])
    
    hdul.writeto(args.output, overwrite=True)
    print(f"MOMFBD deconvolution complete. Output saved to {args.output}")
