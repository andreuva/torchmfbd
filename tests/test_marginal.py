import torchmfbd
from pathlib import Path

def test_joint(make_synthetic_frames):

    n_scans = 1    
    n_obj = 1
    n_frames = 5
    n_x = 32
    n_y = 32

    frames = make_synthetic_frames(n_scans, n_obj, n_frames, n_x, n_y)
    
    patchify = torchmfbd.Patchify4D()    
            
    n_scans, n_obj, n_frames, nx, ny = frames.shape

    config_path = Path(__file__).parent / 'marginal.yaml'
    
    decSI = torchmfbd.Deconvolution(config_path)

    # Patchify and add the frames
    frames_patches = [None] * n_obj
    for i in range(n_obj):
        frames_patches[i] = patchify.patchify(frames[:, i, :, :, :], patch_size=16, stride_size=16, flatten_sequences=True)
        decSI.add_frames(frames_patches[i], id_object=i, id_diversity=0, diversity=0.0)
            
    
    decSI.deconvolve(infer_object=False, 
                     optimizer='adam', 
                     simultaneous_sequences=200,
                     n_iterations=250)    