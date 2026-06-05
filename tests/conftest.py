import pytest
import torch

@pytest.fixture
def make_synthetic_frames():

    def _synthetic_frames(n_scans, n_obj, n_frames, n_x, n_y):
        return torch.rand(n_scans, n_obj, n_frames, n_x, n_y)
    
    return _synthetic_frames