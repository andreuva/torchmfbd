import numpy as np
import os
import h5py
import torch
import matplotlib.pyplot as pl
import sys
sys.path.append('../')
from readsst import readsst
import torchmfbd
from astropy.io import fits


if __name__ == '__main__':

    npix = 256

    obs_file = 'sequence_8542_npix256_aligned.h5'
    print(f'Reading observations from {obs_file}...')
    f = h5py.File(obs_file, 'r')
    im = f['im'][:]
    im_d = None
    f.close()
        
    frames = im[0:2, :, :, 0:npix, 0:npix]
    
    frames /= np.mean(frames, axis=(-1, -2), keepdims=True)

    contrast = np.std(frames, axis=(-1,-2)) / np.mean(frames, axis=(-1,-2))
    ind_best_contrast = np.argmax(contrast[0, 0, :])

    frames = torch.tensor(frames.astype('float32'))

    patchify = torchmfbd.Patchify4D()    
            
    n_scans, n_obj, n_frames, nx, ny = frames.shape
    
    decSI = torchmfbd.Deconvolution('spot_8542_joint.yaml')

    keep_time = False

    if keep_time:
        flatten_sequences = False
    else:
        flatten_sequences = True

    # Patchify and add the frames
    frames_patches = [None] * 2
    for i in range(2):        
        frames_patches[i] = patchify.patchify(frames[:, i, :, :, :], patch_size=92, stride_size=40, flatten_sequences=flatten_sequences)                
        if keep_time:
            decSI.add_frames(frames_patches[i].transpose(0, 1), id_object=i, id_diversity=0, diversity=0.0)
        else:
            decSI.add_frames(frames_patches[i], id_object=i, id_diversity=0, diversity=0.0)
        
    decSI.deconvolve(infer_object=False, 
                     optimizer='adam',                      
                     simultaneous_sequences=8*3,
                     n_iterations=50)
        
    best_frame = []
    obj = []
    for i in range(2):
        if keep_time:
            obj.append(patchify.unpatchify(decSI.obj[i].transpose(0, 1).unsqueeze(2), apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
            best_frame.append(patchify.unpatchify(frames_patches[i][:, :, ind_best_contrast:ind_best_contrast+1, :, :], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
        else:            
            obj.append(patchify.unpatchify(decSI.obj[i][:,None,...], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())        
            best_frame.append(patchify.unpatchify(frames_patches[i][:, ind_best_contrast:ind_best_contrast+1, :, :], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
    

    fig, ax = pl.subplots(ncols=4, nrows=2, figsize=(16, 8), sharex=True, sharey=True)

    for io in range(2):
        for j in range(frames.shape[0]):
            ax[io, 2*j].imshow(best_frame[io][j, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
            ax[io, 2*j+1].imshow(obj[io][j, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)    
    pl.tight_layout()

    print("Updating the cutoffs for the deconvolution...")
    decSI.update_object(cutoffs=[[0.3, 0.35], [0.3, 0.35]])