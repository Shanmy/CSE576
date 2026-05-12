import math
import numpy as np
from scipy.ndimage import maximum_filter
from typing import List
from src.hw1.process_image import get_pixel, set_pixel, copy_image
from src.hw2.modify_image import make_gx_filter, make_gy_filter, convolve_image, l1_normalize
from uwimg import make_image

class Point:
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

class Descriptor:
    def __init__(self):
        self.p = Point()
        self.data = None
        self.n = 0

# Create a feature descriptor for an index in an image.
# image im: source image.
# int i: index in image for the pixel we want to describe.
# returns: descriptor for that index.
def describe_index(im, i: int) -> Descriptor:
    w = 5
    half = w // 2
    d = Descriptor()
    d.p.x = i % im.w
    d.p.y = i // im.w
    d.n = w * w * im.c
    x = i % im.w
    y = i // im.w
    # Pad with edge values to replicate get_pixel clamping behaviour.
    padded = np.pad(im.data, ((0, 0), (half, half), (half, half)), mode='edge')
    patch  = padded[:, y:y + w, x:x + w]           # (c, dy, dx)
    center = im.data[:, y, x][:, None, None]        # (c, 1,  1 )
    # Transpose to (c, dx, dy) so the ravel order matches the original
    # nested loop: for dx in ...: for dy in ...:
    diff   = (center - patch).transpose(0, 2, 1)    # (c, dx, dy)
    d.data = diff.ravel().astype(np.float32)
    return d

# Marks the spot of a point in an image.
# image im: image to mark.
# ponit p: spot to mark in the image.
def mark_spot(im, p: Point) -> None:
    x = int(p.x)
    y = int(p.y)
    for i in range(-9, 10):
        set_pixel(im, x + i, y, 0, 1.0)
        set_pixel(im, x, y + i, 0, 1.0)
        set_pixel(im, x + i, y, 1, 0.0)
        set_pixel(im, x, y + i, 1, 0.0)
        set_pixel(im, x + i, y, 2, 1.0)
        set_pixel(im, x, y + i, 2, 1.0)

# Marks corners denoted by an array of descriptors.
# image im: image to mark.
# descriptor *d: corners in the image.
# int n: number of descriptors to mark.
def mark_corners(im, d: List[Descriptor]) -> None:
    for i in range(len(d)):
        mark_spot(im, d[i].p)

# Creates a 1d Gaussian filter.
# float sigma: standard deviation of Gaussian.
# returns: single row image of the filter.
def make_1d_gaussian(sigma: float):
    w = int(6 * sigma) | 1
    filt = make_image(w, 1, 1)
    center = w // 2
    for x in range(w):
        set_pixel(filt, x, 0, 0, math.exp(-(x - center) ** 2 / (2 * sigma * sigma)))
    l1_normalize(filt)
    return filt

# Smooths an image using separable Gaussian filter.
# image im: image to smooth.
# float sigma: std dev. for Gaussian.
# returns: smoothed image.
def smooth_image(im, sigma: float):
    row = make_1d_gaussian(sigma)
    col = make_image(1, row.w, 1)   # transpose: same weights, oriented vertically
    col.data[0] = row.data[0].T
    return convolve_image(convolve_image(im, row, 1), col, 1)

# Calculate the structure matrix of an image.
# image im: the input image.
# float sigma: std dev. to use for weighted sum.
# returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
#          third channel is IxIy.
def structure_matrix(im, sigma: float):
    S = make_image(im.w, im.h, 3)

    # Compute per-pixel gradients, collapsing channels into one
    Ix = convolve_image(im, make_gx_filter(), 0)
    Iy = convolve_image(im, make_gy_filter(), 0)

    # Element-wise products stored as 1-channel images
    Ix2  = make_image(im.w, im.h, 1)
    Iy2  = make_image(im.w, im.h, 1)
    IxIy = make_image(im.w, im.h, 1)
    Ix2.data[0]  = Ix.data[0] * Ix.data[0]
    Iy2.data[0]  = Iy.data[0] * Iy.data[0]
    IxIy.data[0] = Ix.data[0] * Iy.data[0]

    # Gaussian-weighted sum over neighbourhood (separable)
    S.data[0] = smooth_image(Ix2,  sigma).data[0]
    S.data[1] = smooth_image(Iy2,  sigma).data[0]
    S.data[2] = smooth_image(IxIy, sigma).data[0]

    return S

# Estimate the cornerness of each pixel given a structure matrix S.
# image S: structure matrix for an image.
# returns: a response map of cornerness calculations.
def cornerness_response(S):
    R = make_image(S.w, S.h, 1)
    Ix2  = S.data[0]
    Iy2  = S.data[1]
    IxIy = S.data[2]
    det   = Ix2 * Iy2 - IxIy * IxIy
    trace = Ix2 + Iy2
    R.data[0] = det - 0.06 * trace * trace
    return R

# Perform non-max supression on an image of feature responses.
# image im: 1-channel image of feature responses.
# int w: distance to look for larger responses.
# returns: image with only local-maxima responses within w pixels.
def nms_image(im, w: int):
    r = copy_image(im)
    # maximum_filter with mode='nearest' replicates get_pixel clamping at borders.
    local_max = maximum_filter(im.data[0], size=2 * w + 1, mode='nearest')
    r.data[0] = np.where(im.data[0] == local_max, im.data[0], -999999)
    return r

# Perform harris corner detection and extract features from the corners.
# image im: input image.
# float sigma: std. dev for harris.
# float thresh: threshold for cornerness.
# int nms: distance to look for local-maxes in response map.
# returns: array of descriptors of the corners in the image.
def harris_corner_detector(im, sigma: float, thresh: float, nms: int) -> List[Descriptor]:
    # Calculate structure matrix
    S = structure_matrix(im, sigma)

    # Estimate cornerness
    R = cornerness_response(S)

    # Run NMS on the responses
    Rnms = nms_image(R, nms)

    ys, xs = np.where(Rnms.data[0] > thresh)
    d = [describe_index(im, int(y) * im.w + int(x)) for y, x in zip(ys, xs)]
    return d

# Find and draw corners on an image.
# image im: input image.
# float sigma: std. dev for harris.
# float thresh: threshold for cornerness.
# int nms: distance to look for local-maxes in response map.
def detect_and_draw_corners(im, sigma: float, thresh: float, nms: int) -> None:
    d = harris_corner_detector(im, sigma, thresh, nms)
    mark_corners(im, d)
