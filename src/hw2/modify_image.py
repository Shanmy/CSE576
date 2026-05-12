import math
from typing import List

from uwimg import Image, make_image

from src.hw1.process_image import get_pixel, set_pixel, hsv_to_rgb

TWOPI = 6.2831853


# ----------------------------- Resizing -----------------------------

def nn_interpolate(im: Image, x: float, y: float, c: int) -> float:
    # TODO
    # Performs nearest-neighbor interpolation at floating (x, y) for channel c.
    return get_pixel(im, round(x), round(y), c)


def nn_resize(im: Image, w: int, h: int) -> Image:
    # TODO Fill in (also fix the return line)
    # Uses nearest-neighbor interpolation to resize to (w, h).
    new_im = make_image(w, h, im.c)
    for c in range(im.c):
        for y in range(h):
            for x in range(w):
                new_im.data[c, y, x] = nn_interpolate(im, (x + 0.5) * im.w / w - 0.5, (y + 0.5) * im.h / h - 0.5, c)
    return new_im


def bilinear_interpolate(im: Image, x: float, y: float, c: int) -> float:
    # TODO
    # Performs bilinear interpolation at floating (x, y) for channel c.
    x0 = math.floor(x)
    x1 = x0 + 1
    y0 = math.floor(y)
    y1 = y0 + 1
    fx = x - x0
    fy = y - y0
    return (1 - fx) * (1 - fy) * get_pixel(im, x0, y0, c) \
        + fx * (1 - fy) * get_pixel(im, x1, y0, c) \
        + (1 - fx) * fy * get_pixel(im, x0, y1, c) \
        + fx * fy * get_pixel(im, x1, y1, c)


def bilinear_resize(im: Image, w: int, h: int) -> Image:
    # TODO
    # Uses bilinear interpolation to resize to (w, h).
    new_im = make_image(w, h, im.c)
    for c in range(im.c):
        for y in range(h):
            for x in range(w):
                new_im.data[c, y, x] = bilinear_interpolate(im, (x + 0.5) * im.w / w - 0.5, (y + 0.5) * im.h / h - 0.5, c)
    return new_im


# ----------------------------- Filtering -----------------------------

def l1_normalize(im: Image) -> None:
    # TODO
    # Divide each value by the sum of all values (in-place).
    total = 0
    for c in range(im.c):
        for y in range(im.h):
            for x in range(im.w):
                total += get_pixel(im, x, y, c)
    for c in range(im.c):
        for y in range(im.h):
            for x in range(im.w):
                set_pixel(im, x, y, c, get_pixel(im, x, y, c) / total)


def make_box_filter(w: int) -> Image:
    # TODO
    # Make a (w x w x 1) filter filled with 1s, then l1_normalize.
    filt = make_image(w, w, 1)
    for y in range(w):
        for x in range(w):
            set_pixel(filt, x, y, 0, 1)
    l1_normalize(filt)
    return filt


def convolve_image(im: Image, filt: Image, preserve: int) -> Image:
    # TODO
    # Convolve im with filt. preserve=1 keeps channels, else outputs 1 channel.
    # Must assert (im.c == filt.c) or (filt.c == 1).
    assert im.c == filt.c or filt.c == 1
    new_im = make_image(im.w, im.h, im.c if preserve else 1)
    for c in range(im.c):
        out_c = c if preserve else 0
        filt_c = c if filt.c > 1 else 0
        for y in range(im.h):
            for x in range(im.w):
                val = 0
                for i in range(filt.w):
                    for j in range(filt.h):
                        val += get_pixel(im, x + i - filt.w // 2, y + j - filt.h // 2, c) * get_pixel(filt, i, j, filt_c)
                new_im.data[out_c, y, x] += val
    return new_im


def make_highpass_filter() -> Image:
    filt = make_image(3, 3, 1)
    for y in range(3):
        for x in range(3):
            if x == 1 and y == 1:
                set_pixel(filt, x, y, 0, 4)
            elif x == 1 or y == 1:
                set_pixel(filt, x, y, 0, -1)
    return filt


def make_sharpen_filter() -> Image:
    filt = make_image(3, 3, 1)
    for y in range(3):
        for x in range(3):
            if x == 1 and y == 1:
                set_pixel(filt, x, y, 0, 5)
            elif x == 1 or y == 1:
                set_pixel(filt, x, y, 0, -1)
    return filt


def make_emboss_filter() -> Image:
    kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]
    filt = make_image(3, 3, 1)
    for y in range(3):
        for x in range(3):
            set_pixel(filt, x, y, 0, kernel[y][x])
    return filt


# Question 2.3.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
# Answer: Use preserve=1 for sharpen and emboss, since they operate per-channel to produce a color result.
#          Use preserve=0 for highpass, since it detects edges and the result is a single-channel intensity map.

# Question 2.3.2: Do we have to do any post-processing for the above filters? Which ones and why?
# Answer: Yes. Sharpen, emboss, and highpass can produce values outside [0,1], so clamp_image is needed.
#          Highpass and emboss can go negative; sharpen can exceed 1 on bright edges.


def make_gaussian_filter(sigma: float) -> Image:
    w = int(6 * sigma) | 1
    filt = make_image(w, w, 1)
    center = w // 2
    for y in range(w):
        for x in range(w):
            dx, dy = x - center, y - center
            set_pixel(filt, x, y, 0, math.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma)))
    l1_normalize(filt)
    return filt


def add_image(a: Image, b: Image) -> Image:
    # TODO
    # Assert same shape. Return a+b.
    assert a.w == b.w and a.h == b.h and a.c == b.c
    new_im = make_image(a.w, a.h, a.c)
    for c in range(a.c):
        for y in range(a.h):
            for x in range(a.w):
                set_pixel(new_im, x, y, c, get_pixel(a, x, y, c) + get_pixel(b, x, y, c))
    return new_im


def sub_image(a: Image, b: Image) -> Image:
    # TODO
    # Assert same shape. Return a-b.
    assert a.w == b.w and a.h == b.h and a.c == b.c
    new_im = make_image(a.w, a.h, a.c)
    for c in range(a.c):
        for y in range(a.h):
            for x in range(a.w):
                set_pixel(new_im, x, y, c, get_pixel(a, x, y, c) - get_pixel(b, x, y, c))
    return new_im


def make_gx_filter() -> Image:
    # TODO
    # Create a 3x3 Sobel Gx filter.
    kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    filt = make_image(3, 3, 1)
    for y in range(3):
        for x in range(3):
            set_pixel(filt, x, y, 0, kernel[y][x])
    return filt


def make_gy_filter() -> Image:
    # TODO
    # Create a 3x3 Sobel Gy filter.
    kernel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    filt = make_image(3, 3, 1)
    for y in range(3):
        for x in range(3):
            set_pixel(filt, x, y, 0, kernel[y][x])
    return filt


def feature_normalize(im: Image) -> None:
    # TODO
    # Normalize to [0,1] using (x-min)/(max-min); if max==min set all to 0.
    min_val = float('inf')
    max_val = float('-inf')
    for c in range(im.c):
        for y in range(im.h):
            for x in range(im.w):
                min_val = min(min_val, get_pixel(im, x, y, c))
                max_val = max(max_val, get_pixel(im, x, y, c))
    range_ = max_val - min_val
    for c in range(im.c):
        for y in range(im.h):
            for x in range(im.w):
                set_pixel(im, x, y, c, 0 if range_ == 0 else (get_pixel(im, x, y, c) - min_val) / range_)


def sobel_image(im: Image) -> List[Image]:
    # TODO
    # Return [magnitude, direction] as two 1-channel images.
    gx = convolve_image(im, make_gx_filter(), 0)
    gy = convolve_image(im, make_gy_filter(), 0)
    magnitude = make_image(im.w, im.h, 1)
    direction = make_image(im.w, im.h, 1)
    for y in range(im.h):
        for x in range(im.w):
            set_pixel(magnitude, x, y, 0, math.sqrt(get_pixel(gx, x, y, 0) ** 2 + get_pixel(gy, x, y, 0) ** 2))
            set_pixel(direction, x, y, 0, math.atan2(get_pixel(gy, x, y, 0), get_pixel(gx, x, y, 0)))
    return [magnitude, direction]


def colorize_sobel(im: Image) -> Image:
    magnitude, direction = sobel_image(im)
    feature_normalize(magnitude)
    feature_normalize(direction)
    out = make_image(im.w, im.h, 3)
    for y in range(im.h):
        for x in range(im.w):
            mag = get_pixel(magnitude, x, y, 0)
            set_pixel(out, x, y, 0, get_pixel(direction, x, y, 0))  # H
            set_pixel(out, x, y, 1, mag)                             # S
            set_pixel(out, x, y, 2, mag)                             # V
    hsv_to_rgb(out)
    return out


# EXTRA CREDIT: Median filter
def apply_median_filter(im: Image, kernel_size: int) -> Image:
    new_im = make_image(im.w, im.h, im.c)
    for c in range(im.c):
        for y in range(im.h):
            for x in range(im.w):
                new_im.data[c, y, x] = median_filter(im, x, y, c, kernel_size)
    return new_im

def median_filter(im: Image, x: int, y: int, c: int, kernel_size: int) -> float:
    values = []
    for i in range(kernel_size):
        for j in range(kernel_size):
            values.append(get_pixel(im, x + i - kernel_size // 2, y + j - kernel_size // 2, c))
    return sorted(values)[len(values) // 2]

# SUPER EXTRA CREDIT: Bilateral filter
def apply_bilateral_filter(im: Image, sigma1: float, sigma2: float) -> Image:
    new_im = make_image(im.w, im.h, im.c)
    for c in range(im.c):
        for y in range(im.h):
            for x in range(im.w):
                new_im.data[c, y, x] = bilateral_filter(im, x, y, c, sigma1, sigma2)
    return new_im

def bilateral_filter(im: Image, x: int, y: int, c: int, sigma1: float, sigma2: float) -> float:
    kernel_size = int(6 * sigma1) | 1
    values = []
    for i in range(kernel_size):
        for j in range(kernel_size):
            values.append(get_pixel(im, x + i - kernel_size // 2, y + j - kernel_size // 2, c))
    return sorted(values)[len(values) // 2]
