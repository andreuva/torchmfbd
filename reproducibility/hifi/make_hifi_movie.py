import os
import glob
import argparse
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm

def make_movie(images_dir, pattern, output, fps, max_width):
    search_path = os.path.join(images_dir, pattern)
    frames = sorted(glob.glob(search_path))

    if not frames:
        print(f"No images found matching pattern '{search_path}'.")
        exit(1)

    print(f"Found {len(frames)} frames in {images_dir}. Writing movie to {output} at {fps} fps...")

    # matplotlib's bbox_inches='tight' makes each frame's PNG size vary by a
    # few pixels, so pick one fixed target size (even dims, for H.264) and
    # resize every frame to it, rather than each frame keeping its own size.
    first_size = Image.open(frames[0]).size
    if max_width is not None and first_size[0] > max_width:
        target_w = max_width
        target_h = round(first_size[1] * max_width / first_size[0])
    else:
        target_w, target_h = first_size
    target_w -= target_w % 2
    target_h -= target_h % 2

    # Constant-quality CRF (rather than imageio's variable-bitrate `quality`)
    # and a regular keyframe interval with scene-cut detection disabled, so
    # the encoder doesn't vary quantization/keyframe placement frame to
    # frame in a way that reads as extra flicker on top of the source data.
    writer = imageio.get_writer(
        output, fps=fps, codec='libx264', pixelformat='yuv420p',
        output_params=['-crf', '18', '-g', str(fps * 2), '-sc_threshold', '0'],
    )
    try:
        for frame_path in tqdm(frames, desc="Encoding movie"):
            img = Image.open(frame_path).convert('RGB')
            if img.size != (target_w, target_h):
                img = img.resize((target_w, target_h), Image.LANCZOS)
            writer.append_data(np.asarray(img))
    finally:
        writer.close()

    print(f"Movie saved to {output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assemble MOMFBD reconstructed PNG frames into a movie.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing the PNG frames (output of plot_momfbd_batch_result.py)")
    parser.add_argument("--pattern", type=str, default="*_momfbd.png", help="File matching pattern for frames")
    parser.add_argument("--output", type=str, default=None, help="Output movie path (default: <images_dir>/movie.mp4)")
    parser.add_argument("--fps", type=int, default=18, help="Frames per second")
    parser.add_argument("--max_width", type=int, default=1600, help="Downscale frames wider than this many pixels (None to disable)")
    args = parser.parse_args()

    output = args.output if args.output is not None else os.path.join(args.images_dir, "movie.mp4")
    make_movie(args.images_dir, args.pattern, output, args.fps, args.max_width)
