"""Core imaging physics and restoration methods."""

import numpy as np
from scipy import ndimage
from scipy.signal import wiener

from .constants import VOXEL_SIZE


def project_parallel(volume, axis):
    """Parallel-beam projection using Beer-Lambert law."""
    line_integral = np.sum(volume, axis=axis, dtype=np.float64) * VOXEL_SIZE
    return np.exp(-line_integral).astype(np.float32)


def simulate_acquisition(volume, p):
    """Temporal integration of projections over exposure window."""
    motion_type = p.motion_type

    if motion_type == "none":
        return project_parallel(volume, p.proj_axis)

    total_time = p.exposure_time
    n_steps = p.n_steps
    velocity = p.velocity
    amplitude = p.amplitude
    frequency = p.frequency
    proj_axis = p.proj_axis
    motion_axis = p.motion_axis

    accum = None
    dt = total_time / n_steps

    for i in range(n_steps):
        t = (i + 0.5) * dt
        if motion_type == "linear":
            d_cm = velocity * t
        elif motion_type in ("breathing", "cardiac"):
            d_cm = amplitude * np.sin(2 * np.pi * frequency * t)
        else:
            d_cm = 0.0

        d_vox = d_cm / VOXEL_SIZE
        shift = [0.0, 0.0, 0.0]
        shift[motion_axis] = d_vox

        moved = ndimage.shift(volume, shift, mode="nearest", order=1) if abs(d_vox) > 0.01 else volume
        proj = project_parallel(moved, proj_axis)
        accum = proj if accum is None else accum + proj

    return (accum / n_steps).astype(np.float32)


def add_noise(proj, noise_type, n0):
    """Detector noise models."""
    p = np.clip(proj.copy(), 0.0, 1.0)

    if noise_type == "None":
        return p
    if noise_type == "Poisson":
        counts = np.random.poisson(n0 * p)
        return np.clip(counts / n0, 0, 1).astype(np.float32)
    if noise_type == "Gaussian":
        sigma = 1.0 / np.sqrt(max(n0, 1))
        return np.clip(p + np.random.normal(0, sigma, p.shape), 0, 1).astype(np.float32)
    if noise_type == "Combined":
        counts = np.random.poisson(n0 * p)
        sigma_read = np.sqrt(n0) * 0.01
        read = np.random.normal(0, sigma_read, p.shape)
        return np.clip((counts + read) / n0, 0, 1).astype(np.float32)

    return p


def apply_mitigation(image, method):
    """Post-acquisition mitigation and restoration methods."""
    img = image.copy()

    if method == "None":
        return img
    if method == "Median Filter":
        return ndimage.median_filter(img, size=3).astype(np.float32)
    if method == "Gaussian Smooth":
        return ndimage.gaussian_filter(img, sigma=0.9).astype(np.float32)
    if method == "Wiener Filter":
        return np.clip(wiener(img, mysize=7), 0, 1).astype(np.float32)
    if method == "Unsharp Mask":
        blur = ndimage.gaussian_filter(img, sigma=1.5)
        return np.clip(img + 0.65 * (img - blur), 0, 1).astype(np.float32)
    if method == "RL Deconvolution":
        return _rl_deconv(img, psf=9, iters=12)

    return img


def _rl_deconv(image, psf=9, iters=12):
    psf_k = np.ones((psf, psf), dtype=np.float32) / (psf * psf)
    u = image.copy()
    for _ in range(iters):
        conv = ndimage.convolve(u, psf_k, mode="reflect")
        conv = np.where(conv < 1e-12, 1e-12, conv)
        ratio = image / conv
        u = u * ndimage.convolve(ratio, psf_k[::-1, ::-1], mode="reflect")
    return np.clip(u, 0, 1).astype(np.float32)


def compute_snr(reference, noisy):
    noise = noisy - reference
    rms_s = np.sqrt(np.mean(reference ** 2))
    rms_n = np.sqrt(np.mean(noise ** 2))
    return 20 * np.log10(rms_s / rms_n) if rms_n > 1e-12 else float("inf")


def compute_psnr(reference, noisy):
    mse = np.mean((reference - noisy) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-12 else float("inf")
