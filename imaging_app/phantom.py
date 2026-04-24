"""Phantom generation and body-part cropping."""

import numpy as np
from scipy import ndimage

from .constants import BODY_PARTS, MU


def build_phantom(size=(64, 64, 120)):
    """Construct a 3-D numerical human phantom with attenuation coefficients."""
    nx, ny, nz = size
    ph = np.zeros(size, dtype=np.float32)

    # High-contrast profile: make bone the dominant attenuation class.
    mu_soft = MU["soft_tissue"] * 0.2
    mu_blood = MU["blood"] * 1.2
    mu_cortical = MU["cortical"] * 1.65
    mu_spongy = MU["spongy_bone"] * 1.40

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")

    def ell(cx, cy, cz, rx, ry, rz):
        return ((xg - cx) / rx) ** 2 + ((yg - cy) / ry) ** 2 + ((zg - cz) / rz) ** 2 <= 1.0

    ph[ell(0, 0, 0.05, 0.38, 0.22, 0.50)] = mu_soft
    ph[ell(0, 0, 0.43, 0.46, 0.21, 0.09)] = mu_soft
    ph[ell(0, 0, -0.58, 0.36, 0.19, 0.10)] = mu_soft

    ph[ell(0, 0, 0.60, 0.09, 0.08, 0.07)] = mu_soft

    ph[ell(0, 0, 0.78, 0.20, 0.18, 0.19)] = 0.205
    skull_o = ell(0, 0, 0.78, 0.216, 0.196, 0.206)
    skull_i = ell(0, 0, 0.78, 0.178, 0.158, 0.168)
    ph[skull_o & ~skull_i & (zg > 0.59)] = mu_cortical

    for zv in np.linspace(-0.45, 0.52, 22):
        ph[ell(0.0, -0.10, zv, 0.050, 0.048, 0.024)] = mu_cortical
        ph[ell(0.0, -0.06, zv, 0.035, 0.033, 0.019)] = mu_spongy

    # Rib cage: 4 continuous bands from an elliptical thoracic shell.
    thorax_r = (xg / 0.305) ** 2 + ((yg + 0.015) / 0.175) ** 2
    rib_levels = np.linspace(0.10, 0.40, 4)
    rib_half_thickness = 0.017
    for z0 in rib_levels:
        rib_band = np.abs(zg - z0) <= rib_half_thickness
        cortical_shell = rib_band & (thorax_r <= 1.10) & (thorax_r > 0.88)
        spongy_shell = rib_band & (thorax_r <= 1.00) & (thorax_r > 0.92)
        ph[cortical_shell] = mu_cortical
        ph[spongy_shell] = mu_spongy

    # Emphasize lateral cage contour for a clearer rib-cage outline.
    lateral_outline = (
        (zg >= 0.10)
        & (zg <= 0.40)
        & (thorax_r <= 1.14)
        & (thorax_r > 1.06)
        & (np.abs(xg) > 0.15)
    )
    ph[lateral_outline] = mu_cortical

    # Heart only in thoracic cavity.
    ph[ell(-0.05, 0.02, 0.16, 0.100, 0.090, 0.130)] = mu_blood
    ph[ell(-0.04, 0.01, 0.12, 0.090, 0.082, 0.112)] = mu_blood

    # Two abdominal organs (left/right) with stronger contrast than soft tissue.
    mu_abd_org = MU["blood"] * 1.10
    ph[ell(-0.12, 0.02, -0.20, 0.085, 0.070, 0.095)] = mu_abd_org
    ph[ell(0.12, 0.02, -0.20, 0.085, 0.070, 0.095)] = mu_abd_org

    ph[ell(-0.10, 0.04, -0.12, 0.10, 0.09, 0.08)] = mu_soft
    ph[ell(0.05, 0.03, -0.22, 0.13, 0.10, 0.09)] = mu_soft

    ph[ell(0, 0, -0.62, 0.35, 0.19, 0.12)] = mu_soft
    pbo = ell(0, 0, -0.62, 0.375, 0.215, 0.138)
    pbi = ell(0, 0, -0.62, 0.305, 0.160, 0.102)
    ph[pbo & ~pbi] = mu_spongy

    # Make upper limbs thicker and connected to torso via shoulder overlap.
    for side in (-1, 1):
        shoulder_x = side * 0.34
        ax = side * 0.46
        ph[ell(shoulder_x, 0.00, 0.30, 0.110, 0.090, 0.100)] = mu_soft
        ph[ell(ax, 0.00, 0.16, 0.115, 0.095, 0.300)] = mu_soft
        ph[ell(ax, 0.00, 0.16, 0.048, 0.040, 0.278)] = mu_cortical
        ph[ell(side * 0.49, 0.00, -0.38, 0.095, 0.080, 0.225)] = mu_soft
        ph[ell(side * 0.49, 0.00, -0.38, 0.040, 0.034, 0.205)] = mu_cortical

    for side in (-1, 1):
        lx = side * 0.22
        ph[ell(lx, 0, -0.82, 0.115, 0.095, 0.145)] = mu_soft
        ph[ell(lx, 0, -0.82, 0.044, 0.038, 0.132)] = mu_cortical

    return ph


def crop_phantom(phantom, body_part):
    """Return sub-volume corresponding to selected body part."""
    z_range = BODY_PARTS.get(body_part, (-1.0, 1.0))
    nz = phantom.shape[2]
    z_lin = np.linspace(-1, 1, nz)
    mask = (z_lin >= z_range[0]) & (z_lin <= z_range[1])
    return phantom[:, :, mask]
