import math
import os
from typing import List

import uwimg
from uwimg import (
    Image,
    load_image,
    make_image,
)
from src.hw1.process_image import (
    get_pixel,
    set_pixel,
    rgb_to_grayscale,
    copy_image,
    clamp_image,
    shift_image,
    rgb_to_hsv,
    hsv_to_rgb,
)
from src.hw2.modify_image import (
    nn_interpolate,
    bilinear_interpolate,
    nn_resize,
    bilinear_resize,
    make_highpass_filter,
    make_sharpen_filter,
    make_emboss_filter,
    convolve_image,
    make_box_filter,
    make_gaussian_filter,
    add_image,
    sub_image,
    sobel_image,
)

# Match C constant name
EPS = 0.005
PRINT_PASSES = True

def _gt(path: str) -> str:
    """Return .pillow variant when visionlib is unavailable."""
    if uwimg._vision is None:
        base, ext = os.path.splitext(path)
        alt = base + ".pillow" + ext
        if os.path.exists(alt):
            return alt
    return path

# Hardcoded interpolation expected values — stb originals
_NN_STB = (0.231373, 0.239216, 0.207843, 0.690196)
_BL_STB = (0.231373, 0.237255, 0.206861, 0.678588)

if uwimg._vision is not None:
    _NN_EXP = _NN_STB
    _BL_EXP = _BL_STB
else:
    try:
        from src._pillow_expected import NN_EXPECTED as _NN_EXP, BL_EXPECTED as _BL_EXP
    except ImportError:
        _NN_EXP = _NN_STB
        _BL_EXP = _BL_STB

def free_image(im: Image) -> None:
    # Python GC handles it; keep for parity with C tests.
    return


def avg_diff(a: Image, b: Image) -> float:
    diff = 0.0
    for i in range(a.w * a.h * a.c):
        diff += b.data[i] - a.data[i]
    return diff / (a.w * a.h * a.c)


def center_crop(im: Image) -> Image:
    c = make_image(im.w // 2, im.h // 2, im.c)
    for k in range(im.c):
        for j in range(im.h // 2):
            for i in range(im.w // 2):
                set_pixel(c, i, j, k, get_pixel(im, i + im.w // 4, j + im.h // 4, k))
    return c


def feature_normalize2(im: Image) -> None:
    flat = im.data.reshape(-1)
    mn = float(flat[0])
    mx = float(flat[0])
    for i in range(flat.size):
        v = float(flat[i])
        if v > mx: mx = v
        if v < mn: mn = v

    denom = mx - mn
    if denom == 0.0:
        flat[:] = 0.0
        return
    for i in range(flat.size):
        flat[i] = (float(flat[i]) - mn) / denom


tests_total = 0
tests_fail = 0
_current_test_name = "unknown"


def within_eps(a: float, b: float, eps: float) -> bool:
    return (a - eps) < b and b < (a + eps)


# Print format EXACTLY like C version
def TEST(cond: bool, expr: str) -> None:
    global tests_total, tests_fail
    tests_total += 1
    if not cond:
        tests_fail += 1
        import inspect
        fr = inspect.currentframe().f_back
        rel = os.path.relpath(fr.f_code.co_filename)
        line = fr.f_lineno
        print(f"failed: [{_current_test_name}] testing [{expr}] in {rel}, line {line}")
    else:
        if PRINT_PASSES:
            print(f"passed: [{_current_test_name}] testing [{expr}]")


def same_image(a: Image, b: Image, eps: float) -> bool:
    if a.w != b.w or a.h != b.h or a.c != b.c:
        print("shape mismatch", a.w,a.h,a.c, "vs", b.w,b.h,b.c)
        return False

    ad = a.data.reshape(-1)
    bd = b.data.reshape(-1)
    wh = a.w * a.h

    for idx in range(a.w * a.h * a.c):
        av = float(ad[idx])
        bv = float(bd[idx])

        thresh = (abs(bv) + abs(av)) * eps / 2.0
        ee = eps if thresh <= eps else thresh

        if not within_eps(av, bv, ee):
            c = idx // wh
            rem = idx - c * wh
            y = rem // a.w
            x = rem - y * a.w
            print(f"FAIL idx={idx} (c={c}, y={y}, x={x}) gt={bv} got={av} diff={av-bv}")
            return False
    return True


# ---------------- HOMEWORK 2 ----------------

def test_nn_interpolate() -> None:
    global _current_test_name
    _current_test_name = "test_nn_interpolate"

    im = load_image("data/dogsmall.jpg")
    TEST(within_eps(nn_interpolate(im, -0.5, -0.5, 0), _NN_EXP[0], EPS),
         f"within_eps(nn_interpolate(im, -.5, -.5, 0)  , {_NN_EXP[0]}, EPS)")
    TEST(within_eps(nn_interpolate(im, -0.5, 0.5, 1), _NN_EXP[1], EPS),
         f"within_eps(nn_interpolate(im, -.5, .5, 1)   , {_NN_EXP[1]}, EPS)")
    TEST(within_eps(nn_interpolate(im, 0.499, 0.5, 2), _NN_EXP[2], EPS),
         f"within_eps(nn_interpolate(im, .499, .5, 2)  , {_NN_EXP[2]}, EPS)")
    TEST(within_eps(nn_interpolate(im, 14.2, 15.9, 1), _NN_EXP[3], EPS),
         f"within_eps(nn_interpolate(im, 14.2, 15.9, 1), {_NN_EXP[3]}, EPS)")
    free_image(im)


def test_bl_interpolate() -> None:
    global _current_test_name
    _current_test_name = "test_bl_interpolate"

    im = load_image("data/dogsmall.jpg")
    TEST(within_eps(bilinear_interpolate(im, -0.5, -0.5, 0), _BL_EXP[0], EPS),
         f"within_eps(bilinear_interpolate(im, -.5, -.5, 0)  , {_BL_EXP[0]}, EPS)")
    TEST(within_eps(bilinear_interpolate(im, -0.5, 0.5, 1), _BL_EXP[1], EPS),
         f"within_eps(bilinear_interpolate(im, -.5, .5, 1)   , {_BL_EXP[1]}, EPS)")
    TEST(within_eps(bilinear_interpolate(im, 0.499, 0.5, 2), _BL_EXP[2], EPS),
         f"within_eps(bilinear_interpolate(im, .499, .5, 2)  , {_BL_EXP[2]}, EPS)")
    TEST(within_eps(bilinear_interpolate(im, 14.2, 15.9, 1), _BL_EXP[3], EPS),
         f"within_eps(bilinear_interpolate(im, 14.2, 15.9, 1), {_BL_EXP[3]}, EPS)")
    free_image(im)


def test_nn_resize() -> None:
    global _current_test_name
    _current_test_name = "test_nn_resize"

    im = load_image("data/dogsmall.jpg")
    resized = nn_resize(im, im.w * 4, im.h * 4)
    gt = load_image(_gt("figs/dog4x-nn-for-test.png"))
    TEST(same_image(resized, gt, EPS),
         "same_image(resized, gt, EPS)")
    free_image(im)
    free_image(resized)
    free_image(gt)

    im2 = load_image("data/dog.jpg")
    resized2 = nn_resize(im2, 713, 467)
    gt2 = load_image(_gt("figs/dog-resize-nn.png"))
    TEST(same_image(resized2, gt2, EPS),
         "same_image(resized2, gt2, EPS)")
    free_image(im2)
    free_image(resized2)
    free_image(gt2)


def test_bl_resize() -> None:
    global _current_test_name
    _current_test_name = "test_bl_resize"

    im = load_image("data/dogsmall.jpg")
    resized = bilinear_resize(im, im.w * 4, im.h * 4)
    gt = load_image(_gt("figs/dog4x-bl.png"))
    TEST(same_image(resized, gt, EPS),
         "same_image(resized, gt, EPS)")
    free_image(im)
    free_image(resized)
    free_image(gt)

    im2 = load_image("data/dog.jpg")
    resized2 = bilinear_resize(im2, 713, 467)
    gt2 = load_image(_gt("figs/dog-resize-bil.png"))
    TEST(same_image(resized2, gt2, EPS),
         "same_image(resized2, gt2, EPS)")
    free_image(im2)
    free_image(resized2)
    free_image(gt2)


def test_multiple_resize() -> None:
    global _current_test_name
    _current_test_name = "test_multiple_resize"

    im = load_image("data/dog.jpg")
    for _ in range(10):
        im1 = bilinear_resize(im, im.w * 4, im.h * 4)
        im2 = bilinear_resize(im1, im1.w // 4, im1.h // 4)
        free_image(im)
        free_image(im1)
        im = im2
    gt = load_image(_gt("figs/dog-multipleresize.png"))
    TEST(same_image(im, gt, EPS),
         "same_image(im, gt, EPS)")
    free_image(im)
    free_image(gt)


def test_highpass_filter() -> None:
    global _current_test_name
    _current_test_name = "test_highpass_filter"

    im = load_image("data/dog.jpg")
    f = make_highpass_filter()
    blur = convolve_image(im, f, 0)
    clamp_image(blur)

    gt = load_image(_gt("figs/dog-highpass.png"))
    TEST(same_image(blur, gt, EPS),
         "same_image(blur, gt, EPS)")
    free_image(im)
    free_image(f)
    free_image(blur)
    free_image(gt)


def test_emboss_filter() -> None:
    global _current_test_name
    _current_test_name = "test_emboss_filter"

    im = load_image("data/dog.jpg")
    f = make_emboss_filter()
    blur = convolve_image(im, f, 1)
    clamp_image(blur)

    gt = load_image(_gt("figs/dog-emboss.png"))
    TEST(same_image(blur, gt, EPS),
         "same_image(blur, gt, EPS)")
    free_image(im)
    free_image(f)
    free_image(blur)
    free_image(gt)


def test_sharpen_filter() -> None:
    global _current_test_name
    _current_test_name = "test_sharpen_filter"

    im = load_image("data/dog.jpg")
    f = make_sharpen_filter()
    blur = convolve_image(im, f, 1)
    clamp_image(blur)

    gt = load_image(_gt("figs/dog-sharpen.png"))
    TEST(same_image(blur, gt, EPS),
         "same_image(blur, gt, EPS)")
    free_image(im)
    free_image(f)
    free_image(blur)
    free_image(gt)


def test_convolution() -> None:
    global _current_test_name
    _current_test_name = "test_convolution"

    im = load_image("data/dog.jpg")
    f = make_box_filter(7)
    blur = convolve_image(im, f, 1)
    clamp_image(blur)

    gt = load_image(_gt("figs/dog-box7.png"))
    TEST(same_image(blur, gt, EPS),
         "same_image(blur, gt, EPS)")
    free_image(im)
    free_image(f)
    free_image(blur)
    free_image(gt)


def test_gaussian_filter() -> None:
    global _current_test_name
    _current_test_name = "test_gaussian_filter"

    f = make_gaussian_filter(7)
    f.data *= 100

    gt = load_image("figs/gaussian_filter_7.png")
    TEST(same_image(f, gt, EPS),
         "same_image(f, gt, EPS)")
    free_image(f)
    free_image(gt)


def test_gaussian_blur() -> None:
    global _current_test_name
    _current_test_name = "test_gaussian_blur"

    im = load_image("data/dog.jpg")
    f = make_gaussian_filter(2)
    blur = convolve_image(im, f, 1)
    clamp_image(blur)

    gt = load_image(_gt("figs/dog-gauss2.png"))
    TEST(same_image(blur, gt, EPS),
         "same_image(blur, gt, EPS)")
    free_image(im)
    free_image(f)
    free_image(blur)
    free_image(gt)


def test_hybrid_image() -> None:
    global _current_test_name
    _current_test_name = "test_hybrid_image"

    melisa = load_image("data/melisa.png")
    aria = load_image("data/aria.png")
    f = make_gaussian_filter(2)
    lfreq_m = convolve_image(melisa, f, 1)
    lfreq_a = convolve_image(aria, f, 1)
    hfreq_a = sub_image(aria, lfreq_a)
    reconstruct = add_image(lfreq_m, hfreq_a)
    gt = load_image("figs/hybrid.png")
    clamp_image(reconstruct)
    TEST(same_image(reconstruct, gt, EPS),
         "same_image(reconstruct, gt, EPS)")
    free_image(melisa)
    free_image(aria)
    free_image(f)
    free_image(lfreq_m)
    free_image(lfreq_a)
    free_image(hfreq_a)
    free_image(reconstruct)
    free_image(gt)


def test_frequency_image() -> None:
    global _current_test_name
    _current_test_name = "test_frequency_image"

    im = load_image("data/dog.jpg")
    f = make_gaussian_filter(2)
    lfreq = convolve_image(im, f, 1)
    hfreq = sub_image(im, lfreq)
    reconstruct = add_image(lfreq, hfreq)

    low_freq = load_image(_gt("figs/low-frequency.png"))
    high_freq = load_image(_gt("figs/high-frequency-clamp.png"))

    clamp_image(lfreq)
    clamp_image(hfreq)
    TEST(same_image(lfreq, low_freq, EPS),
         "same_image(lfreq, low_freq, EPS)")
    TEST(same_image(hfreq, high_freq, EPS),
         "same_image(hfreq, high_freq, EPS)")
    TEST(same_image(reconstruct, im, EPS),
         "same_image(reconstruct, im, EPS)")
    free_image(im)
    free_image(f)
    free_image(lfreq)
    free_image(hfreq)
    free_image(reconstruct)
    free_image(low_freq)
    free_image(high_freq)


def test_sobel() -> None:
    global _current_test_name
    _current_test_name = "test_sobel"

    im = load_image("data/dog.jpg")
    res = sobel_image(im)
    mag = res[0]
    theta = res[1]
    feature_normalize2(mag)
    feature_normalize2(theta)

    gt_mag = load_image(_gt("figs/magnitude.png"))
    gt_theta = load_image(_gt("figs/theta.png"))

    TEST(gt_mag.w == mag.w and gt_theta.w == theta.w,
         "gt_mag.w == mag.w && gt_theta.w == theta.w")
    TEST(gt_mag.h == mag.h and gt_theta.h == theta.h,
         "gt_mag.h == mag.h && gt_theta.h == theta.h")
    TEST(gt_mag.c == mag.c and gt_theta.c == theta.c,
         "gt_mag.c == mag.c && gt_theta.c == theta.c")

    if (gt_mag.w != mag.w or gt_theta.w != theta.w or
        gt_mag.h != mag.h or gt_theta.h != theta.h or
        gt_mag.c != mag.c or gt_theta.c != theta.c):
        return

    gt_mag_flat   = gt_mag.data.reshape(-1)
    gt_theta_flat = gt_theta.data.reshape(-1)
    mag_flat      = mag.data.reshape(-1)
    theta_flat    = theta.data.reshape(-1)

    for i in range(gt_mag.w * gt_mag.h):
        if within_eps(float(gt_mag_flat[i]), 0.0, EPS):
            gt_theta_flat[i] = 0.0
            theta_flat[i] = 0.0
        if within_eps(float(gt_theta_flat[i]), 0.0, EPS) or within_eps(float(gt_theta_flat[i]), 1.0, EPS):
            gt_theta_flat[i] = 0.0
            theta_flat[i] = 0.0

    TEST(same_image(mag, gt_mag, EPS),
         "same_image(mag, gt_mag, EPS)")
    TEST(same_image(theta, gt_theta, EPS),
         "same_image(theta, gt_theta, EPS)")
    free_image(im)
    free_image(mag)
    free_image(theta)
    free_image(gt_mag)
    free_image(gt_theta)


def test_hw2() -> None:
    # EXACT order from C
    test_nn_interpolate()
    test_nn_resize()
    test_bl_interpolate()
    test_bl_resize()
    # test_multiple_resize()
    test_gaussian_filter()
    test_sharpen_filter()
    test_emboss_filter()
    test_highpass_filter()
    test_convolution()
    test_gaussian_blur()
    test_hybrid_image()
    test_frequency_image()
    test_sobel()
    print(f"{tests_total} tests, {tests_total - tests_fail} passed, {tests_fail} failed")