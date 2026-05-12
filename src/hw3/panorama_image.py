import math
import random
import numpy as np
from functools import cmp_to_key
from typing import List
from src.hw1.process_image import get_pixel, set_pixel, copy_image
from src.hw2.modify_image import bilinear_interpolate
from uwimg import Image, make_image, save_image
from src.hw3.harris_image import Point, Descriptor, harris_corner_detector, mark_corners

class Match:
    def __init__(self):
        self.p = Point()
        self.q = Point()
        self.ai = 0
        self.bi = 0
        self.distance = 0.0

# Comparator for matches
# Match a, b: objects to compare.
# returns: result of comparison, 0 if same, 1 if a > b, -1 if a < b.
def match_compare(a: Match, b: Match) -> int:
    if a.distance < b.distance: return -1
    elif a.distance > b.distance: return 1
    else: return 0

# Helper function to create 2d points.
# float x, y: coordinates of point.
# returns: the point.
def make_point(x: float, y: float) -> Point:
    p = Point()
    p.x = x
    p.y = y
    return p

# Place two images side by side on canvas, for drawing matching pixels.
# image a, b: images to place.
# returns: image with both a and b side-by-side.
def both_images(a: Image, b: Image):
    both = make_image(a.w + b.w, max(a.h, b.h), max(a.c, b.c))
    both.data[:a.c, :a.h, :a.w] = a.data
    both.data[:b.c, :b.h, a.w:a.w + b.w] = b.data
    return both

# Draws lines between matching pixels in two images.
# image a, b: two images that have matches.
# List[Match] matches: list of matches between a and b.
# int n: number of matches.
# int inliers: number of inliers at beginning of matches, drawn in green.
# returns: image with matches drawn between a and b on same canvas.
def draw_matches(a: Image, b: Image, matches: List[Match], n: int, inliers: int):
    both = both_images(a, b)
    for i in range(n):
        bx = int(matches[i].p.x)
        ex = int(matches[i].q.x)
        by = int(matches[i].p.y)
        ey = int(matches[i].q.y)
        for j in range(bx, ex + a.w):
            r = int(float(j - bx) / (ex + a.w - bx) * (ey - by) + by)
            set_pixel(both, j, r, 0, 0.0 if i < inliers else 1.0)
            set_pixel(both, j, r, 1, 1.0 if i < inliers else 0.0)
            set_pixel(both, j, r, 2, 0.0)
    return both

# Draw the matches with inliers in green between two images.
# image a, b: two images to match.
# List[Match] m: matches list
def draw_inliers(a: Image, b: Image, H: np.ndarray, m: List[Match], n: int, thresh: float):
    inliers = model_inliers(H, m, n, thresh)
    lines = draw_matches(a, b, m, n, inliers)
    return lines

# Find corners, match them, and draw them between two images.
# image a, b: images to match.
# float sigma: gaussian for harris corner detector. Typical: 2
# float thresh: threshold for corner/no corner. Typical: 1-5
# int nms: window to perform nms on. Typical: 3
def find_and_draw_matches(a: Image, b: Image, sigma: float, thresh: float, nms: int):
    mn = [0]
    ad = harris_corner_detector(a, sigma, thresh, nms)
    bd = harris_corner_detector(b, sigma, thresh, nms)
    m = match_descriptors(ad, len(ad), bd, len(bd), mn)

    mark_corners(a, ad)
    mark_corners(b, bd)
    lines = draw_matches(a, b, m, mn[0], 0)
    return lines

# Calculates L1 distance between to floating point arrays.
# float array a, b: arrays to compare.
# int n: number of values in each array.
# returns: l1 distance between arrays (sum of absolute differences).
def l1_distance(a: np.ndarray, b: np.ndarray, n: int) -> float:
    return float(np.sum(np.abs(a[:n] - b[:n])))

# Finds best matches between descriptors of two images.
# List[Descriptor] a, b: list of descriptors for pixels in two images.
# int an, bn: number of descriptors in arrays a and b.
# List[int] mn: single-element list to hold number of matches found.
# returns: best matches found. each descriptor in a should match with at most
#          one other descriptor in b.
def match_descriptors(a: List[Descriptor], an: int, b: List[Descriptor], bn: int, mn: List[int]) -> List[Match]:
    mn[0] = an
    m = [Match() for _ in range(an)]

    # Vectorised L1 distance matrix: shape (an, bn).
    A = np.stack([a[j].data[:a[j].n] for j in range(an)])          # (an, n)
    B = np.stack([b[i].data[:b[i].n] for i in range(bn)])          # (bn, n)
    dists = np.sum(np.abs(A[:, np.newaxis, :] - B[np.newaxis, :, :]), axis=2)  # (an, bn)
    bind_arr   = np.argmin(dists, axis=1)                           # (an,)
    best_dists = dists[np.arange(an), bind_arr]                     # (an,)

    for j in range(an):
        bind = int(bind_arr[j])
        m[j].ai = j
        m[j].bi = bind
        m[j].p  = a[j].p
        m[j].q  = b[bind].p
        m[j].distance = float(best_dists[j])

    count = 0
    seen = [0] * bn
    m.sort(key=cmp_to_key(match_compare))
    for i in range(an):
        if seen[m[i].bi] == 0:
            seen[m[i].bi] = 1
            m[count] = m[i]
            count += 1
    mn[0] = count
    return m

# Apply a projective transformation to a point.
# np.ndarray H: homography to project point.
# point p: point to project.
# returns: point projected using the homography.
def project_point(H: np.ndarray, p: Point) -> Point:
    v = H @ np.array([p.x, p.y, 1.0])
    return make_point(v[0] / v[2], v[1] / v[2])

# Calculate L2 distance between two points.
# point p, q: points.
# returns: L2 distance between them.
def point_distance(p: Point, q: Point) -> float:
    return math.sqrt((p.x - q.x)**2 + (p.y - q.y)**2)

# Count number of inliers in a set of matches. Should also bring inliers
# to the front of the array.
# np.ndarray H: homography between coordinate systems.
# List[Match] m: matches to compute inlier/outlier.
# int n: number of matches in m.
# float thresh: threshold to be an inlier.
# returns: number of inliers whose projected point falls within thresh of
#          their match in the other image. Should also rearrange matches
#          so that the inliers are first in the array. For drawing.
def model_inliers(H: np.ndarray, m: List[Match], n: int, thresh: float) -> int:
    if n == 0:
        return 0
    # Vectorised projection of all p points through H.
    pts  = np.array([[m[i].p.x, m[i].p.y, 1.0] for i in range(n)], dtype=np.float64)
    proj = (H @ pts.T).T                          # (n, 3)
    px   = proj[:, 0] / proj[:, 2]
    py   = proj[:, 1] / proj[:, 2]
    qx   = np.array([m[i].q.x for i in range(n)])
    qy   = np.array([m[i].q.y for i in range(n)])
    inlier_mask = np.sqrt((px - qx) ** 2 + (py - qy) ** 2) < thresh

    count = 0
    for i in range(n):
        if inlier_mask[i]:
            m[count], m[i] = m[i], m[count]
            count += 1
    return count

# Randomly shuffle matches for RANSAC.
# List[Match] m: matches to shuffle in place.
# int n: number of elements in matches.
def randomize_matches(m: List[Match], n: int) -> None:
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        m[i], m[j] = m[j], m[i]

# Computes homography between two images given matching pixels.
# List[Match] matches: matching points between images.
# int n: number of matches to use in calculating homography.
# returns: matrix representing homography H that maps image a to image b.
def compute_homography(matches: List[Match], n: int) -> np.ndarray:
    M = np.zeros((n * 2, 8), dtype=np.float64)
    b = np.zeros((n * 2, 1), dtype=np.float64)

    xs  = np.array([matches[i].p.x for i in range(n)])
    xps = np.array([matches[i].q.x for i in range(n)])
    ys  = np.array([matches[i].p.y for i in range(n)])
    yps = np.array([matches[i].q.y for i in range(n)])

    zeros = np.zeros(n)
    ones  = np.ones(n)
    M[0::2] = np.column_stack([xs,    ys,    ones,  zeros, zeros, zeros, -xs * xps, -ys * xps])
    M[1::2] = np.column_stack([zeros, zeros, zeros, xs,    ys,    ones,  -xs * yps, -ys * yps])
    b[0::2, 0] = xps
    b[1::2, 0] = yps

    a, res, rank, sv = np.linalg.lstsq(M, b, rcond=None)
    if rank < 8:
        return None

    H = np.array([[a[0,0], a[1,0], a[2,0]],
                  [a[3,0], a[4,0], a[5,0]],
                  [a[6,0], a[7,0], 1.0   ]])
    return H

# Perform RANdom SAmple Consensus to calculate homography for noisy matches.
# List[Match] m: set of matches.
# int n: number of matches.
# float thresh: inlier/outlier distance threshold.
# int k: number of iterations to run.
# int cutoff: inlier cutoff to exit early.
# returns: matrix representing most common homography between matches.
def RANSAC(m: List[Match], n: int, thresh: float, k: int, cutoff: int) -> np.ndarray:
    best = 0
    Hb = np.eye(3, dtype=np.float64)
    Hb[0, 2] = 256.0
    Hb[1, 2] = 0.0
    for _ in range(k):
        randomize_matches(m, n)
        H = compute_homography(m, 4)
        if H is None:
            continue
        inliers = model_inliers(H, m, n, thresh)
        if inliers > best:
            H = compute_homography(m, inliers)
            if H is None:
                continue
            best = model_inliers(H, m, n, thresh)
            Hb = H
            if best >= cutoff:
                return Hb
    return Hb

# Stitches two images together using a projective transformation.
# image a, b: images to stitch.
# np.ndarray H: homography from image a coordinates to image b coordinates.
# returns: combined image stitched together.
def combine_images(a: Image, b: Image, H: np.ndarray):
    Hinv = np.linalg.inv(H)

    # Project the corners of image b into image a coordinates.
    c1 = project_point(Hinv, make_point(0, 0))
    c2 = project_point(Hinv, make_point(b.w - 1, 0))
    c3 = project_point(Hinv, make_point(0, b.h - 1))
    c4 = project_point(Hinv, make_point(b.w - 1, b.h - 1))

    # Find top left and bottom right corners of image b warped into image a.
    botright = make_point(max(c1.x, c2.x, c3.x, c4.x), max(c1.y, c2.y, c3.y, c4.y))
    topleft  = make_point(min(c1.x, c2.x, c3.x, c4.x), min(c1.y, c2.y, c3.y, c4.y))

    # Find how big our new image should be and the offsets from image a.
    dx = int(min(0, topleft.x))
    dy = int(min(0, topleft.y))
    w  = int(max(a.w, botright.x) - dx)
    h  = int(max(a.h, botright.y) - dy)

    # Can disable this if you are making very big panoramas.
    # Usually this means there was an error in calculating H.
    if w > 7000 or h > 7000:
        print("output too big, stopping")
        return copy_image(a)

    c = make_image(w, h, a.c)

    # Paste image a into the canvas at offset (-dx, -dy).
    c.data[:a.c, -dy:-dy + a.h, -dx:-dx + a.w] = a.data

    # Build a grid of all canvas pixel coordinates and project into b's space.
    ii, jj = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))             # both (h, w)
    coords = np.stack([(ii + dx).ravel(),
                       (jj + dy).ravel(),
                       np.ones(w * h)], axis=0)                      # (3, w*h)
    proj = H @ coords                                                 # (3, w*h)
    qx   = proj[0] / proj[2]
    qy   = proj[1] / proj[2]

    valid  = (qx >= 0) & (qx < b.w) & (qy >= 0) & (qy < b.h)
    qx_v, qy_v = qx[valid], qy[valid]

    x0 = np.floor(qx_v).astype(int)
    y0 = np.floor(qy_v).astype(int)
    x1 = np.minimum(x0 + 1, b.w - 1)
    y1 = np.minimum(y0 + 1, b.h - 1)
    x0 = np.maximum(x0, 0)
    y0 = np.maximum(y0, 0)
    fx = qx_v - np.floor(qx_v)
    fy = qy_v - np.floor(qy_v)

    ci = ii.ravel()[valid].astype(int)
    cj = jj.ravel()[valid].astype(int)

    for k in range(b.c):
        c.data[k, cj, ci] = (b.data[k, y0, x0] * (1 - fx) * (1 - fy) +
                              b.data[k, y0, x1] * fx        * (1 - fy) +
                              b.data[k, y1, x0] * (1 - fx) * fy        +
                              b.data[k, y1, x1] * fx        * fy)
    return c

# Create a panoramam between two images.
# image a, b: images to stitch together.
# float sigma: gaussian for harris corner detector. Typical: 2
# float thresh: threshold for corner/no corner. Typical: 1-5
# int nms: window to perform nms on. Typical: 3
# float inlier_thresh: threshold for RANSAC inliers. Typical: 2-5
# int iters: number of RANSAC iterations. Typical: 1,000-50,000
# int cutoff: RANSAC inlier cutoff. Typical: 10-100
# int draw: flag to draw inliers.
def panorama_image(a: Image, b: Image, sigma: float = 2, thresh: float = 5, nms: int = 3, inlier_thresh: float = 3, iters: int = 10000, cutoff: int = 100, draw: int = 0):
    random.seed(10)
    mn = [0]

    # Calculate corners and descriptors
    ad = harris_corner_detector(a, sigma, thresh, nms)
    bd = harris_corner_detector(b, sigma, thresh, nms)

    # Find matches
    m = match_descriptors(ad, len(ad), bd, len(bd), mn)

    # Run RANSAC to find the homography
    H = RANSAC(m, mn[0], inlier_thresh, iters, cutoff)

    if draw:
        # Mark corners and matches between images
        mark_corners(a, ad)
        mark_corners(b, bd)
        inlier_matches = draw_inliers(a, b, H, m, mn[0], inlier_thresh)
        save_image(inlier_matches, "output/inliers")

    # Stitch the images together with the homography
    comb = combine_images(a, b, H)
    return comb

# Project an image onto a cylinder.
# image im: image to project.
# float f: focal length used to take image (in pixels).
# returns: image projected onto cylinder, then flattened.
def cylindrical_project(im: Image, f: float):
    c  = make_image(im.w, im.h, im.c)
    cx = im.w / 2.0
    cy = im.h / 2.0

    ii, jj = np.meshgrid(np.arange(im.w, dtype=np.float64),
                         np.arange(im.h, dtype=np.float64))   # both (h, w)
    theta  = (ii - cx) / f
    x_flat = f * np.tan(theta) + cx
    y_flat = (jj - cy) / np.cos(theta) + cy

    valid  = (x_flat >= 0) & (x_flat < im.w) & (y_flat >= 0) & (y_flat < im.h)
    xf_v   = x_flat[valid]
    yf_v   = y_flat[valid]

    x0 = np.floor(xf_v).astype(int)
    y0 = np.floor(yf_v).astype(int)
    x1 = np.minimum(x0 + 1, im.w - 1)
    y1 = np.minimum(y0 + 1, im.h - 1)
    x0 = np.maximum(x0, 0)
    y0 = np.maximum(y0, 0)
    fx = xf_v - np.floor(xf_v)
    fy = yf_v - np.floor(yf_v)

    ci = ii[valid].astype(int)
    cj = jj[valid].astype(int)

    for k in range(im.c):
        c.data[k, cj, ci] = (im.data[k, y0, x0] * (1 - fx) * (1 - fy) +
                              im.data[k, y0, x1] * fx        * (1 - fy) +
                              im.data[k, y1, x0] * (1 - fx) * fy        +
                              im.data[k, y1, x1] * fx        * fy)
    return c
