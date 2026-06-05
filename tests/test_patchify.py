import pytest
import torchmfbd
import torch

def test_patchify():

    n_scans = 1    
    n_frames = 2
    n_x = 80
    n_y = 80

    frames = torch.randn(n_scans, n_frames, n_x, n_y)

    p = torchmfbd.Patchify4D()

    patches = p.patchify(frames, patch_size=20, stride_size=10, flatten_sequences=True)
    out = p.unpatchify(patches, apodization=0)

    assert frames.shape == out.shape, f'Expected {frames.shape}, got {out.shape}'
    
    patches = p.patchify(frames, patch_size=20, stride_size=10, flatten_sequences=False)        
    out = p.unpatchify(patches, apodization=0)

    assert frames.shape == out.shape, f'Expected {frames.shape}, got {out.shape}'