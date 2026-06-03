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
    sigma = 0.01
    modes = [5, 9, 14, 20, 27, 35, 44, 65, 90, 119]
    n_modes = len(modes)
    n_iter = 350

    for r0 in r0s:

        f = h5py.File(f'nmodes_marginal_r0_{r0}.h5', 'w')

        imgs = np.load(f'convolved_r0_{r0}.npz')
        frames = imgs['convolved'][None, :, :, :]

        frames /= np.mean(frames, axis=(-1, -2), keepdims=True)

        frames += sigma * np.random.normal(size=frames.shape)

        contrast = np.std(frames, axis=(-1,-2)) / np.mean(frames, axis=(-1,-2))
        ind_best_contrast = np.argmax(contrast[0, :])

        frames = torch.tensor(frames.astype('float32'))
        
        for i, n_m in enumerate(modes):
                                    
            ff = open('marginal.yaml', 'r')
            lines = ff.readlines()            
            ff.close()
            lines[33] = f'    nmax_modes : {n_m}\n'
            ff = open('tmp.yaml', 'w')
            ff.writelines(lines)
            ff.close()
                        
            decSI = torchmfbd.Deconvolution('tmp.yaml')

            
            decSI.add_frames(frames, id_object=0, id_diversity=0, diversity=0.0)
                            
            decSI.deconvolve(infer_object=False, 
                            optimizer='adam', 
                            simultaneous_sequences=200,
                            n_iterations=int(n_iter * (n_m / modes[0])))
                        
            best_frame = frames[:, ind_best_contrast, :, :].cpu().numpy()[0, ...]
            obj = decSI.obj[0].cpu().numpy()[0, ...]

            nx, ny = best_frame.shape
                
            # Save the object as an HDF5 file
            if i == 0:
                dbest_frame = f.create_dataset(f'best_frame', shape=(n_modes, nx, ny), dtype='float32')
                dobj = f.create_dataset(f'obj', shape=(n_modes, nx, ny), dtype='float32')     
                dloss = f.create_dataset(f'loss', shape=(n_modes), dtype='float32')


            dbest_frame[i, ...] = best_frame
            dobj[i, ...] = obj
            dloss[i, ...] = decSI.loss.cpu().numpy()[-1]

        f.close()