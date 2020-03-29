import argparse
import os
from functools import reduce, partial
from itertools import accumulate

import cv2
import math
import numpy as np
import scipy
from scipy.signal import convolve2d
import scipy.stats as st
from scipy.ndimage.filters import gaussian_filter


def generate_kernel(dim, sigma=0.4):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-sigma, sigma, dim + 1)
    kern1d = st.norm.pdf(x)
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


def reduce_img(image, factor=2, kernel_size=5):
    """reduce image by 1/2"""
    kernel = generate_kernel(kernel_size)
    outimage = scipy.signal.convolve2d(image, kernel, 'same')
    out = outimage[::factor, ::factor]
    return out


def expand_img(image, factor=2, kernel_size=5):
    """expand image by a factor"""
    kernel = generate_kernel(kernel_size)
    h, w = image.shape
    outimage = np.zeros((h * factor, w * factor),
                        dtype=np.float64)
    outimage[::factor, ::factor] = image[:, :]
    out = factor ** 2 * scipy.signal.convolve2d(outimage, kernel, 'same')
    return out


def gauss_pyramid(image, levels):
    """create a gaussain pyramid of a given image"""
    return list(accumulate([image] * levels, lambda a, _: reduce_img(a)))


def consolidate(img1, img2):
    """Consolidate images to the same size in case of small offsets"""
    for i in range(2):
        if img1.shape[i] > img2.shape[i]:
            img1 = np.delete(img1, (-1), axis=i)
    return img1, img2


def lapl_pyramid(gauss_pyr):
    """build a laplacian pyramid"""
    output = []
    for i in range(len(gauss_pyr) - 1):
        gaussian = gauss_pyr[i]
        expanded = expand_img(gauss_pyr[i + 1])
        expanded, gaussian = consolidate(expanded, gaussian)
        output.append(gaussian - expanded)
    output.append(gauss_pyr[-1])
    return output


def blend_lapl_pyramids(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
    """Blend the two laplacian pyramids by weighting them according to the
    mask.
    """

    def blend_layer(white, black, mask):
        return mask * white + (1 - mask) * black

    return [blend_layer(*layers) for layers in
            zip(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask)]


def collapse_pyramid(lapl_pyr):
    """Reconstruct the image based on its laplacian pyramid."""

    def collapse(big_layer, small_layer):
        expanded = expand_img(big_layer)
        return sum(consolidate(expanded, small_layer))

    return reduce(collapse, lapl_pyr[::-1])


def gauss_pyramid_bgr(image):
    bgr = cv2.split(image)
    bgr = [c.astype(float) for c in bgr]
    min_size = min(bgr[0].shape)
    # at least 16x16 at the highest level.
    depth = int(math.floor(math.log(min_size, 2))) - 4
    return [gauss_pyramid(c, depth) for c in bgr]


def lapl_pyramid_bgr(pyramid):
    return [lapl_pyramid(c) for c in pyramid]


def blend_images(images, mask):
    def bound(img):
        img[img < 0] = 0
        img[img > 255] = 255
        return img.astype(np.uint8)

    gauss_mask = gauss_pyramid_bgr(mask)
    lapl_pyramids = [lapl_pyramid_bgr(gauss_pyramid_bgr(img))
                     for img in images]

    collapsed = [
            collapse_pyramid(blend_lapl_pyramids(lapl1, lapl2, gauss_mask)) for
            lapl1, lapl2, gauss_mask in zip(*lapl_pyramids, gauss_mask)]

    collapsed = [bound(c) for c in collapsed]

    return np.dstack(np.array(collapsed))


def make_mask(dimensions, split):
    h, w, d = dimensions
    split = split if split <= w else w
    split = split if split >= 0 else 0
    left = np.zeros((h, split, d))
    right = np.full((h, w - split, d), 1)
    return np.concatenate((left, right), axis=1)


def move_split(dimensions, split, direction, speed):
    direction = 1 if direction else - 1
    split = split + direction * speed
    return make_mask(dimensions, split), split


def process_img(images, split=None, direction=True, speed=1):
    dimensions = h, w, d = images[0].shape
    split = w // 2 if split is None else split
    mask, split = move_split(dimensions, split, direction, speed)
    img = blend_images(images, mask)
    return img, True, split


def save(images, split, out='blended'):
    img, running, split = process_img(images, split)
    file_count = 0
    while True:
        file_name = '{}_{}.png'.format(out, file_count)
        if os.path.isfile(file_name):
            file_count += 1
            continue
        cv2.imwrite(file_name, img)
        print('Saved image to {}'.format(file_name))
        return img, running, split


def close(images, split):
    return images, False, split


def main(filenames, outname):
    images = [cv2.imread(fn) for fn in filenames]

    for img, filename in zip(images, filenames):
        if img is None:
            print('Unable to open image at {}'.format(filename))
            return

    # check if images are the same size
    img_shapes = [img.shape for img in images]
    have_same_dims = lambda a, b: all(a[i] == b[i] for i in range(len(a)))
    if not reduce(have_same_dims, img_shapes):
        print('Images must be the same resolution! Instead got:')
        for filename, size in zip(filenames, img_shapes):
            print(
                    'Image {} has size: {}x{}'.format(filename, size[0],
                                                      size[1]))
        return

    slow = 3
    fast = 6
    right = True
    left = False
    keymap = {
            ord('s'): partial(process_img, direction=left, speed=slow),
            ord('a'): partial(process_img, direction=left, speed=fast),
            ord('d'): partial(process_img, direction=right, speed=slow),
            ord('f'): partial(process_img, direction=right, speed=fast),
            ord('m'): process_img,
            ord('w'): partial(save, out=outname),
            ord('q'): close,
    }

    img, running, split = process_img(images)

    running = True
    while running:
        cv2.imshow('Blended image', img)
        k = cv2.waitKey(0)
        if k in keymap.keys():
            split = None if k == ord('m') else split
            img, running, split = keymap[k](images, split)
        else:
            print('Unassigned key {}'.format(chr(k)))


def parse_arguments():
    desc = 'Image blending software for MWR classes.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('images', nargs=2,
                        help='Paths to images for blending. As of now the '
                             'program only supports the blending of two '
                             'images. These can be separate names or a shell '
                             'wildcard pattern.')
    parser.add_argument('-o', '--out', default='blended',
                        help='First part of output file name. If empty, '
                             'the image will be named "blended"')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args.images, args.out)
