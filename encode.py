#!/usr/bin/env python3

import argparse
import time
from io import BytesIO
import os
from os.path import isdir, isfile
import glob
from PIL import Image
import numpy as np
from shutil import copyfile

import pframe_dataset_shared


# Putting a .jpg in the extension means we can directly look at the results.
# They are JPGs after all.
EXTENSION = 'baseline.jpg'


# Very low to get low bpp.
JPG_QUALITY = 7


def encoder(frame1, frame2):
    # Convert to long so that the subtraction does not overflow
    residual_normalized = (frame2.astype(np.long) - frame1) // 2 + 127
    # Convert back to uint8
    residual_normalized = residual_normalized.astype(np.uint8)
    f = BytesIO()
    # optimize=True optimizes Huffman tables.
    Image.fromarray(residual_normalized).save(f, format='jpeg', quality=JPG_QUALITY, optimize=True)

    return f.getvalue()


def compress_folder(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    animation_dirs = [d for d in os.listdir(data_dir) if isdir(os.path.join(data_dir, d))]
    for animation_num, d in enumerate(animation_dirs):
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)
        inputs = sorted(glob.glob(os.path.join(os.path.join(data_dir, d), '*.png')))
        assert len(inputs) > 0, 'No inputs!'

        N = len(inputs)
        num_channels = 3
        start = time.time()

        bpps = []  # Bpps of individual images
        total_bytes = 0  # Of all files
        bytes_img = []  # Store bytes of Y, U, V

        for count, p1 in enumerate(inputs):
            p2 = ''
            if count + num_channels < len(inputs):
                p2 = inputs[count + num_channels]
            if isfile(p2):
                if count < num_channels:
                    p_out = os.path.join(os.path.join(output_dir, d), os.path.splitext(os.path.basename(p1))[0] + '.png')
                    copyfile(p1, p_out)

                p_out = os.path.join(os.path.join(output_dir, d), os.path.splitext(os.path.basename(p2))[0] + '.' + EXTENSION)
                i1, i2 = np.array(Image.open(p1)), np.array(Image.open(p2))

                encoded = encoder(i1, i2)
                bytes_img.append(len(encoded))
                if '_y.png' in p1:  # Y always comes last!
                    assert len(bytes_img) == num_channels, len(bytes_img)
                    total_bytes += sum(bytes_img)
                    bpp = sum(bytes_img) * 8 / np.prod(i1.shape)
                    bpps.append(bpp)
                    bytes_img = []
                with open(p_out, 'wb') as f_out:
                    f_out.write(encoded)

                if count > 0 and count % 50 == 0:
                    elapsed = time.time() - start
                    per_img = elapsed / count
                    remaining = (N - count) * per_img
                    print(('\rAnimation {}/{}. Q={}: Wrote {}/{} files. Time: {:.1f}s // '
                           '{:.3e} per img // {:.3f} bpp, {} bytes // '
                           '~{:.1f}s remaining').format(
                              animation_num + 1, len(animation_dirs),
                              JPG_QUALITY, count, N, elapsed, 
                              per_img, np.mean(bpps), int(total_bytes), 
                              remaining), end='', flush=True)
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data_dir', help="Directory of data.")
    p.add_argument('output_dir')
    flags = p.parse_args()

    compress_folder(os.path.expanduser(flags.data_dir),
                    os.path.expanduser(flags.output_dir))


if __name__ == '__main__':
    main()
