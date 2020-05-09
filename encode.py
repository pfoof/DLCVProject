#!/usr/bin/env python3

"""
Simple non-learned baseline.

ENCODE: (function `encoder` below)
  Inputs: frame 1 (F1) and frame 2 (F2).
  Encode F2 given F1:
  1. Calculate Residual_Normalized = (F2 - F1) // 2 + 127
     (this is now in range {0, ..., 255}.
      The idea of the //2 is to have 256 possible values, because
      otherwise we would have 511 values.)
  2. Compress Residual_Normalized with JPG
  -> Save to Bitstream
DECODE: (function `decoder` below)
  Inputs: F1 and Bitstream
  1. Get Residual_Normalized from JPG in Bistream
  2. F2' = F1 + ( Residual_normalized - 127 ) * 2
     (F2' is the reconstruction)
"""

import argparse
import time
from io import BytesIO
import os
from os.path import isdir, isfile
import glob
from PIL import Image
import numpy as np

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
        start = time.time()

        bpps = []  # Bpps of individual images
        total_bytes = 0  # Of all files
        bytes_img = []  # Store bytes of Y, U, V

        for count, p1 in enumerate(inputs):
            p2 = p1.split('_')
            p2[-2] = str(int(p2[-2]) + 1)
            while len(p2[-2]) < 5:
                p2[-2] = '0' + p2[-2]
            p2 = '_'.join(p2)
            #print(p2)
            if isfile(p2):
                p2_expected = pframe_dataset_shared.get_frame_path(p1, offset=1)
                assert os.path.basename(p2_expected) == os.path.basename(p2), (p1, p2)
                i1, i2 = np.array(Image.open(p1)), np.array(Image.open(p2))

                p_out = os.path.join(os.path.join(output_dir, d), os.path.splitext(os.path.basename(p1))[0] + '.' + EXTENSION)
                encoded = encoder(i1, i2)
                bytes_img.append(len(encoded))
                if '_y.png' in p1:  # Y always comes last!
                    assert len(bytes_img) == 3, len(bytes_img)
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
