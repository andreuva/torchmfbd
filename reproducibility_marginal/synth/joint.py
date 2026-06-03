import numpy as np
import os
import h5py
import torch
import matplotlib.pyplot as pl
import sys
sys.path.append('../')
from readsst import readsst
import torchmfbd
import h5py


if __name__ == '__main__':

    r0s = [10, 15, 20, 30]
    noise = [0.001, 0.01, 0.1]
    n_noise = len(noise)

    for r0 in r0s:

        f = h5py.File(f'joint_r0_{r0}.h5', 'w')
        
        for i, sigma in enumerate(noise):
            imgs = np.load(f'convolved_r0_{r0}.npz')
            frames = imgs['convolved'][None, :, :, :]

            frames /= np.mean(frames, axis=(-1, -2), keepdims=True)

            frames += sigma * np.random.normal(size=frames.shape)
            
            contrast = np.std(frames, axis=(-1,-2)) / np.mean(frames, axis=(-1,-2))
            ind_best_contrast = np.argmax(contrast[0, :])

            frames = torch.tensor(frames.astype('float32'))
                        
            decSI = torchmfbd.Deconvolution('joint.yaml')

            
            decSI.add_frames(frames, id_object=0, id_diversity=0, diversity=0.0)
                            
            decSI.deconvolve(infer_object=False, 
                            optimizer='adam', 
                            simultaneous_sequences=200,
                            n_iterations=250)
                        
            best_frame = frames[:, ind_best_contrast, :, :].cpu().numpy()[0, ...]
            obj = decSI.obj[0].cpu().numpy()[0, ...]

            nx, ny = best_frame.shape
                
            # Save the object as an HDF5 file
            if i == 0:
                dbest_frame = f.create_dataset(f'best_frame', shape=(n_noise, nx, ny), dtype='float32')
                dobj = f.create_dataset(f'obj', shape=(n_noise, nx, ny), dtype='float32')                

            dbest_frame[i, ...] = best_frame
            dobj[i, ...] = obj

        f.close()