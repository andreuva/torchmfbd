import numpy as np
import matplotlib.pyplot as pl
import h5py
import matplotlib.animation as animation

f = h5py.File('reconstruction_original.h5', 'r')
n_t = f['obj_0'].shape[0]
orig_0 = f['obj_0'][:]
orig_1 = f['obj_1'][:]
best_0 = f['best_frame_0'][:]
best_1 = f['best_frame_1'][:]
f.close()

regu0 = []
regu1 = []
f = h5py.File('reconstruction_regularized_1e-4.h5', 'r')
regu0.append(f['obj_0'][:])
regu1.append(f['obj_1'][:])
f.close()

f = h5py.File('reconstruction_regularized_contrast_3e-4.h5', 'r')
regu0.append(f['obj_0'][:])
regu1.append(f['obj_1'][:])
f.close()

f = h5py.File('reconstruction_regularized_1e-3.h5', 'r')
regu0.append(f['obj_0'][:])
regu1.append(f['obj_1'][:])
f.close()

im = [None] * 16

fig, ax = pl.subplots(ncols=4, nrows=4, figsize=(10, 10), sharex=True, sharey=True)

im[0] = ax[0, 0].imshow(best_0[0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
im[1] = ax[0, 1].imshow(orig_0[0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
im[2] = ax[0, 2].imshow(best_1[0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
im[3] = ax[0, 3].imshow(orig_1[0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)

im[4] = ax[1, 0].imshow(best_0[0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
im[5] = ax[1, 1].imshow(regu0[0][0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
im[6] = ax[1, 2].imshow(best_1[0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
im[7] = ax[1, 3].imshow(regu1[0][0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)

im[8] = ax[2, 0].imshow(best_0[0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
im[9] = ax[2, 1].imshow(regu0[1][0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
im[10] = ax[2, 2].imshow(best_1[0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
im[11] = ax[2, 3].imshow(regu1[1][0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)

im[12] = ax[3, 0].imshow(best_0[0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
im[13] = ax[3, 1].imshow(regu0[2][0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
im[14] = ax[3, 2].imshow(best_1[0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)
im[15] = ax[3, 3].imshow(regu1[2][0, 0, :, :], cmap='gray', vmin=0.5, vmax=1.5)

ax[0, 0].set_title('No regularization')
ax[1, 0].set_title('Regularized (1e-4)')
ax[2, 0].set_title('Regularized (3e-4)')
ax[3, 0].set_title('Regularized (1e-3)')

def update(frame):
    im[0].set_array(best_0[frame, 0, :, :])
    im[1].set_array(orig_0[frame, 0, :, :])
    im[2].set_array(best_1[frame, 0, :, :])
    im[3].set_array(orig_1[frame, 0, :, :])

    im[4].set_array(best_0[frame, 0, :, :])
    im[5].set_array(regu0[0][frame, 0, :, :])
    im[6].set_array(best_1[frame, 0, :, :])
    im[7].set_array(regu1[0][frame, 0, :, :])

    im[8].set_array(best_0[frame, 0, :, :])
    im[9].set_array(regu0[1][frame, 0, :, :])
    im[10].set_array(best_1[frame, 0, :, :])
    im[11].set_array(regu1[1][frame, 0, :, :])

    im[12].set_array(best_0[frame, 0, :, :])
    im[13].set_array(regu0[2][frame, 0, :, :])
    im[14].set_array(best_1[frame, 0, :, :])
    im[15].set_array(regu1[2][frame, 0, :, :])

    return im

ani = animation.FuncAnimation(fig, 
                                  update, 
                                  frames=n_t, 
                                  interval=100, 
                                  blit=True)

# 5. Save the animation as a GIF
# Note: This requires the 'pillow' library installed (`pip install pillow`)
output_filename = f"movie.mp4"
writer = animation.FFMpegWriter(fps=6, metadata=dict(artist='Me'), bitrate=1800)
ani.save(output_filename, writer=writer)
