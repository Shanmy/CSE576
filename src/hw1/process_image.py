import math
from uwimg import Image, make_image


def get_pixel(im: Image, x: int, y: int, c: int) -> float:
    x = max(0, min(x, im.w - 1))
    y = max(0, min(y, im.h - 1))
    return im.data[c, y, x]


def set_pixel(im: Image, x: int, y: int, c: int, v: float) -> None:
    x = max(0, min(x, im.w - 1))
    y = max(0, min(y, im.h - 1))
    im.data[c, y, x] = v
    return


def copy_image(im: Image) -> Image:
    copy = make_image(im.w, im.h, im.c)
    copy.data = im.data.copy()
    return copy


def rgb_to_grayscale(im: Image) -> Image:
    assert im.c == 3
    gray = make_image(im.w, im.h, 1)
    for y in range(im.h):
        for x in range(im.w):
            r = get_pixel(im, x, y, 0)
            g = get_pixel(im, x, y, 1)
            b = get_pixel(im, x, y, 2)
            gray.data[0, y, x] = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


def shift_image(im: Image, c: int, v: float) -> None:
    for y in range(im.h):
        for x in range(im.w):
            set_pixel(im, x, y, c, get_pixel(im, x, y, c) + v)
    return


def clamp_image(im: Image) -> None:
    for y in range(im.h):
        for x in range(im.w):
            for c in range(im.c):
                set_pixel(im, x, y, c, max(0, min(get_pixel(im, x, y, c), 1)))
    return


# These might be handy
def three_way_max(a: float, b: float, c: float) -> float:
    return (a if a > b else b) if ((a if a > b else b) > c) else c


def three_way_min(a: float, b: float, c: float) -> float:
    return (a if a < b else b) if ((a if a < b else b) < c) else c


def rgb_to_hsv(im: Image) -> None:
    for y in range(im.h):
        for x in range(im.w):
            r = get_pixel(im, x, y, 0)
            g = get_pixel(im, x, y, 1)
            b = get_pixel(im, x, y, 2)
            v = max(r, g, b)
            c = v - min(r, g, b)
            s = c / v if v != 0 else 0
            if s == 0:
                h = 0
            elif v == r:
                h = (g - b) / c
            elif v == g:
                h = 2 + (b - r) / c
            else:
                h = 4 + (r - g) / c
            h /= 6
            if h < 0:
                h += 1
            im.data[0, y, x] = h
            im.data[1, y, x] = s
            im.data[2, y, x] = v
    return


def hsv_to_rgb(im: Image) -> None:
    for y in range(im.h):
        for x in range(im.w):
            h = get_pixel(im, x, y, 0)
            s = get_pixel(im, x, y, 1)
            v = get_pixel(im, x, y, 2)
            c = v * s
            hp = h * 6
            x_ = c * (1 - abs(hp % 2 - 1))
            m = v - c
            i = int(hp)
            if i == 0:   r, g, b = c, x_, 0
            elif i == 1: r, g, b = x_, c, 0
            elif i == 2: r, g, b = 0, c, x_
            elif i == 3: r, g, b = 0, x_, c
            elif i == 4: r, g, b = x_, 0, c
            else:        r, g, b = c, 0, x_
            im.data[0, y, x] = r + m
            im.data[1, y, x] = g + m
            im.data[2, y, x] = b + m
    return