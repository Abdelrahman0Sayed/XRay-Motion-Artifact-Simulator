"""Phantom generation and body-part cropping."""

import numpy as np
from scipy import ndimage

from .constants import BODY_PARTS, MU

#TODO: make a more convenient phantom 
def build_phantom(size=(64, 64, 120)):
    """Construct a 3-D numerical human phantom with attenuation coefficients."""
    nx, ny, nz = size
    ph = np.zeros(size, dtype=np.float32)

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")

    def ell(cx, cy, cz, rx, ry, rz):
        return ((xg - cx) / rx) ** 2 + ((yg - cy) / ry) ** 2 + ((zg - cz) / rz) ** 2 <= 1.0

    ph[ell(0, 0, 0.05, 0.38, 0.22, 0.50)] = MU["soft_tissue"]
    ph[ell(0, 0, 0.43, 0.46, 0.21, 0.09)] = MU["soft_tissue"]
    ph[ell(0, 0, -0.58, 0.36, 0.19, 0.10)] = MU["soft_tissue"]

    ph[ell(0, 0, 0.60, 0.09, 0.08, 0.07)] = MU["soft_tissue"]

    ph[ell(0, 0, 0.78, 0.20, 0.18, 0.19)] = 0.205
    skull_o = ell(0, 0, 0.78, 0.216, 0.196, 0.206)
    skull_i = ell(0, 0, 0.78, 0.190, 0.170, 0.180)
    ph[skull_o & ~skull_i & (zg > 0.59)] = MU["cortical"]

    for zv in np.linspace(-0.45, 0.52, 22):
        ph[ell(0.0, -0.10, zv, 0.042, 0.040, 0.022)] = MU["cortical"]
        ph[ell(0.0, -0.06, zv, 0.030, 0.028, 0.018)] = MU["spongy_bone"]

    for zv in np.linspace(0.05, 0.43, 10):
        for side in (-1, 1):
            for theta in np.linspace(0.05, np.pi - 0.05, 20):
                rx = side * 0.30 * np.sin(theta)
                ry = -0.08 + 0.17 * np.cos(theta)
                ph[ell(rx, ry, zv, 0.032, 0.028, 0.024)] = MU["cortical"]

    ph[ell(-0.17, 0.02, 0.22, 0.125, 0.090, 0.250)] = MU["lung"]
    ph[ell(0.17, 0.02, 0.22, 0.125, 0.090, 0.250)] = MU["lung"]

    ph[ell(-0.04, 0.01, 0.12, 0.090, 0.082, 0.112)] = MU["blood"]
    ph[ell(0.13, 0.05, -0.08, 0.185, 0.125, 0.105)] = MU["liver"]

    ph[ell(-0.10, 0.04, -0.12, 0.10, 0.09, 0.08)] = MU["soft_tissue"]
    ph[ell(0.05, 0.03, -0.22, 0.13, 0.10, 0.09)] = MU["soft_tissue"]

    ph[ell(0, 0, -0.62, 0.35, 0.19, 0.12)] = MU["soft_tissue"]
    pbo = ell(0, 0, -0.62, 0.375, 0.215, 0.138)
    pbi = ell(0, 0, -0.62, 0.305, 0.160, 0.102)
    ph[pbo & ~pbi] = MU["spongy_bone"]

    for side in (-1, 1):
        ax = side * 0.60
        ph[ell(ax, 0, 0.18, 0.072, 0.062, 0.285)] = MU["soft_tissue"]
        ph[ell(ax, 0, 0.18, 0.026, 0.024, 0.265)] = MU["cortical"]
        ph[ell(ax, 0, -0.38, 0.063, 0.055, 0.195)] = MU["soft_tissue"]

    for side in (-1, 1):
        lx = side * 0.22
        ph[ell(lx, 0, -0.82, 0.115, 0.095, 0.145)] = MU["soft_tissue"]
        ph[ell(lx, 0, -0.82, 0.036, 0.032, 0.125)] = MU["cortical"]

    return ph


def crop_phantom(phantom, body_part):
    """Return sub-volume corresponding to selected body part."""
    z_range = BODY_PARTS.get(body_part, (-1.0, 1.0))
    nz = phantom.shape[2]
    z_lin = np.linspace(-1, 1, nz)
    mask = (z_lin >= z_range[0]) & (z_lin <= z_range[1])
    return phantom[:, :, mask]
