"""Core imaging physics and restoration methods."""

import numpy as np
from scipy import ndimage
from skimage.restoration import richardson_lucy
from .constants import BODY_PARTS, VOXEL_SIZE


# ---------------------------------------------------------------------------
# Beer-Lambert parallel-beam projection
# ---------------------------------------------------------------------------

def project_parallel(volume, axis):
    """
    Parallel-beam X-ray projection using Beer-Lambert law.

    Computes line-integral attenuation along the specified axis and applies
    the exponential attenuation model: I = I0 * exp(-∫μ(s)ds), where the
    integral is the cumulative linear attenuation coefficient along the ray.

    Physics
    -------
    This implements the standard Beer-Lambert law for X-ray transmission through
    matter. The volume contains linear attenuation coefficients (μ), typically
    in units of cm⁻¹. The line integral ∫μ(s)ds represents the optical depth;
    exponentiating the negative line integral gives the transmission fraction
    (intensity relative to incident intensity I0).

    Parameters
    ----------
    volume : ndarray, shape (nx, ny, nz)
        3-D array of linear attenuation coefficients [cm⁻¹]. Must be non-negative.
    axis : int
        Projection axis (0=Lateral, 1=AP/PA). Defines the ray direction.

    Returns
    -------
    proj : ndarray, shape (nx, ny) or permutation thereof
        2-D transmission image with values in [0, 1], where 0 = fully attenuated,
        1 = no attenuation. Output dtype is float32.

    Notes
    -----
    - The line integral is computed by summing along the specified axis and
      multiplying by VOXEL_SIZE (voxel dimension in cm) to convert from
      discrete sum to physical distance integral.
    - Output values saturating at 0 indicate very high attenuation depth.
    """
    line_integral = np.sum(volume, axis=axis, dtype=np.float64) * VOXEL_SIZE
    return np.exp(-line_integral).astype(np.float32)


# ---------------------------------------------------------------------------
# Anatomical motion weight mask
# ---------------------------------------------------------------------------

def _motion_weight_vector(nz: int, body_part: str, motion_type: str) -> np.ndarray:
    """
    Build a 1-D weight array  w[k], k = 0 .. nz-1, that scales the applied
    displacement amplitude for each z-slice of the (already-cropped) volume.

    Physics basis
    -------------
    Respiratory motion is driven by the diaphragm and confined to the thoracic
    cavity.  Cardiac motion is strictly pericardial.  Structures below the
    diaphragm (pelvis, legs) show < 0.5 mm displacement from either source.

    References
    ----------
    Keall et al., Med. Phys. 33(10):3874-3900, 2006  (respiratory motion)
    Shechter et al., IEEE TMI 23(8):1046-1056, 2004  (cardiac motion)

    Phantom coordinate system:  z in [-1, 1],  -1 = feet,  +1 = skull crown
      Thorax (chest)   : z in [0.00, 0.57]
      Heart silhouette : z in [0.05, 0.45]
      Diaphragm-abdomen coupling : z in [-0.20, 0.00]  (exponential decay)
    """
    z_lo, z_hi = BODY_PARTS.get(body_part, (-1.0, 1.0))
    z_lin = np.linspace(z_lo, z_hi, nz)      # global phantom z for each slice

    w = np.zeros(nz, dtype=np.float64)

    if motion_type == "breathing":
        # Full weight inside thorax; exponential decay into upper abdomen.
        THORAX_LO   =  0.00
        THORAX_HI   =  0.57
        ABD_LO      = -0.20      # diaphragm coupling zone lower bound
        DECAY_K     =  8.0       # decay rate in z-units (~cm⁻¹ equivalent)

        for k, z in enumerate(z_lin):
            if THORAX_LO <= z <= THORAX_HI:
                w[k] = 1.0
            elif ABD_LO <= z < THORAX_LO:
                w[k] = np.exp(-DECAY_K * (THORAX_LO - z))
            # below ABD_LO and above THORAX_HI: w[k] = 0.0

    else:
        # Linear motion or unknown: whole body moves uniformly.
        w[:] = 1.0

    return w.astype(np.float64)


# ---------------------------------------------------------------------------
# Spatially-weighted volume shift
# ---------------------------------------------------------------------------

def _shift_with_mask(volume: np.ndarray, d_vox: float,
                     motion_axis: int, w_z: np.ndarray) -> np.ndarray:
    """
    Shift a volume with a z-slice-dependent weight using inverse coordinate
    mapping (map_coordinates), so each destination voxel is pulled from its
    weighted source position.

    For Z-axis motion (breathing / cardiac — always axis 2 in this app):
        source_z[k] = k  −  w_z[k] * d_vox
    This is a true per-slice displacement with no Python loop overhead.

    For X-axis motion (linear lateral — axis 0):
        Each z-slice is shifted in X by  w_z[k] * d_vox  independently.
    """
    nx, ny, nz = volume.shape

    if motion_axis == 2:
        # ------------------------------------------------------------------
        # Z-axis inverse coordinate mapping  (vectorised, no Python loop)
        # ------------------------------------------------------------------
        k_dst = np.arange(nz, dtype=np.float64)
        k_src = k_dst - w_z * d_vox                  # shape (nz,)

        xi = np.arange(nx, dtype=np.float64)[:, None, None]
        yi = np.arange(ny, dtype=np.float64)[None, :, None]
        zi = k_src[None, None, :]                     # broadcast over (nx, ny, nz)

        xi = np.broadcast_to(xi, (nx, ny, nz)).ravel()
        yi = np.broadcast_to(yi, (nx, ny, nz)).ravel()
        zi = np.broadcast_to(zi, (nx, ny, nz)).ravel()

        coords = np.array([xi, yi, zi])
        moved  = ndimage.map_coordinates(
            volume, coords, order=1, mode="nearest"
        ).reshape(nx, ny, nz)
        return moved.astype(np.float32)

    elif motion_axis == 0:
        # ------------------------------------------------------------------
        # X-axis per-slice shift  (lateral motion, only non-zero slices)
        # ------------------------------------------------------------------
        moved = volume.copy()
        for k in range(nz):
            sx = w_z[k] * d_vox
            if abs(sx) > 0.01:
                moved[:, :, k] = ndimage.shift(
                    volume[:, :, k], [sx, 0.0], mode="nearest", order=1
                )
        return moved.astype(np.float32)

    else:
        # Fallback: uniform shift on whichever axis was requested.
        shift = [0.0, 0.0, 0.0]
        shift[motion_axis] = d_vox
        return ndimage.shift(
            volume, shift, mode="nearest", order=1
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# Main acquisition simulation
# ---------------------------------------------------------------------------

def simulate_acquisition(volume, p):
    """
    Temporal integration of projections over the X-ray exposure window.

    The detected intensity is the time-average of Beer-Lambert projections:

        I_det(u,v) = (1/T) ∫₀ᵀ I(u, v, t) dt

    This integral is approximated by STRATIFIED MONTE CARLO sampling:
    one random sample per sub-interval  [i·dt, (i+1)·dt].

        t_i = (i + U[0,1]) · dt,   i = 0 … N-1

    Why stratified random instead of fixed midpoint?
    ------------------------------------------------
    The midpoint rule  t_i = (i + 0.5)·dt  integrates linear functions
    exactly for ANY N — so N=5 and N=100 produced identical results for
    linear motion.  Stratified sampling breaks this degeneracy:

      • Low  N  → high variance per run → grainy, rough motion blur
      • High N  → low variance          → smooth, converged motion blur

    This is also physically motivated: photon arrival times are Poisson-
    distributed, not deterministic.

    Anatomical motion boundary (Task 1 fix)
    ----------------------------------------
    For breathing / cardiac motion, a physics-based z-weight vector w_z
    confines the displacement to the thoracic cavity.  Structures outside
    the thorax are either attenuated or entirely stationary, matching the
    literature (Keall 2006; Shechter 2004).
    """
    motion_type = p.motion_type

    if motion_type == "none":
        return project_parallel(volume, p.proj_axis)

    total_time  = p.exposure_time
    n_steps     = p.n_steps
    velocity    = p.velocity
    amplitude   = p.amplitude
    frequency   = p.frequency
    proj_axis   = p.proj_axis
    motion_axis = p.motion_axis
    body_part   = p.body_part

    # Build the anatomical z-weight mask (computed once, outside the loop).
    nz  = volume.shape[2]
    w_z = _motion_weight_vector(nz, body_part, motion_type)

    # Only use the mask when at least one slice has partial weight
    # (i.e., the mask is non-trivial and the motion is sinusoidal).
    use_mask = (motion_type == "breathing") and (not np.all(w_z == 1.0))

    accum = None
    dt    = total_time / n_steps

    for i in range(n_steps):
        # ------------------------------------------------------------------
        # Stratified random sample within sub-interval i
        # ------------------------------------------------------------------
        t = (i + np.random.uniform(0.0, 1.0)) * dt

        if motion_type == "linear":
            d_cm = velocity * t
        elif motion_type == "breathing":
            d_cm = amplitude * np.sin(2.0 * np.pi * frequency * t)
        else:
            d_cm = 0.0

        d_vox = d_cm / VOXEL_SIZE

        if abs(d_vox) > 0.01:
            if use_mask:
                moved = _shift_with_mask(volume, d_vox, motion_axis, w_z)
            else:
                shift = [0.0, 0.0, 0.0]
                shift[motion_axis] = d_vox
                moved = ndimage.shift(volume, shift, mode="nearest", order=1)
        else:
            moved = volume

        proj  = project_parallel(moved, proj_axis)
        accum = proj if accum is None else accum + proj

    return (accum / n_steps).astype(np.float32)


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------

def add_noise(proj, noise_type, n0):
    """Detector noise models.

    Poisson  — quantum (photon-counting) noise:  N ~ Poisson(N₀·p)
    Gaussian — electronic read noise:  σ ∝ 1/√N₀
    Combined — both sources summed (realistic flat-panel model)
    """
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
        counts     = np.random.poisson(n0 * p)
        sigma_read = np.sqrt(n0) * 0.01
        read       = np.random.normal(0, sigma_read, p.shape)
        return np.clip((counts + read) / n0, 0, 1).astype(np.float32)

    return p


# ---------------------------------------------------------------------------
# Mitigation / restoration
# ---------------------------------------------------------------------------

def apply_mitigation(image, p):
    """Post-acquisition mitigation and restoration methods."""
    img = image.copy()
    method = p.mitigation

    if method == "None":
        return img
    if method == "Unsharp Mask":
        blur = ndimage.gaussian_filter(img, sigma=1.5)
        return np.clip(img + 0.65 * (img - blur), 0, 1).astype(np.float32)
    if method == "RL Deconvolution":
        return _rl_deconv(img, p, iters=15)

    return img


def _rl_deconv(image, p, iters=15):
    """Richardson-Lucy deconvolution with exact PSF matching and shift realignment."""
    if p.motion_type == "none":
        return image

    # 1. Calculate physical blur extent and build the correct kernel shape
    if p.motion_type == "linear":
        total_d = p.velocity * p.exposure_time
        blur_pixels = max(3, int(total_d / VOXEL_SIZE))
        # Linear motion creates a UNIFORM flat blur, not a curved window
        window = np.ones(blur_pixels, dtype=np.float32)
    else:
        total_d = 2.0 * p.amplitude
        blur_pixels = max(3, int(total_d / VOXEL_SIZE))
        # Sinusoidal motion requires a curved window
        window = np.hanning(blur_pixels + 2)[1:-1].astype(np.float32)

    window /= window.sum()
    
    if p.motion_axis == 2:
        psf_k = window.reshape(1, -1)
    else:
        psf_k = window.reshape(-1, 1)

    # 2. Pad the image matrix to prevent boundary ringing
    pad_w = blur_pixels * 2
    img_padded = np.pad(image, pad_width=pad_w, mode="edge")

    # 3. Built-in Richardson-Lucy Iterations
    u_padded = richardson_lucy(img_padded, psf_k, num_iter=iters, clip=False)

    # 4. Crop the padding back off
    u = u_padded[pad_w:-pad_w, pad_w:-pad_w]

# 5. Phase Shift Realignment (Crucial for accurate SSIM/NMSE)
    # The forward simulation shifted the anatomy forward. The blur is centered at +blur_pixels/2.
    # We must shift the deconvolved image backward to align with the unshifted static reference.
    shift_amount = -blur_pixels / 2.0
    shift_vec = [0.0, 0.0]
    
    # Map the 3D motion axis to the corresponding 2D projection axis
    if p.motion_axis == 2:
        shift_vec[1] = shift_amount  # 3D Z-axis maps to 2D axis 1
    else:
        shift_vec[0] = shift_amount  # 3D X-axis maps to 2D axis 0
    
    # Use order=1 (bilinear) for accurate sub-pixel shifting
    u_aligned = ndimage.shift(u, shift_vec, mode="nearest", order=1)

    # 6. Spatially-Aware Masking
    nz = image.shape[1]
    w_z = _motion_weight_vector(nz, p.body_part, p.motion_type)
    mask_2d = np.tile(w_z, (image.shape[0], 1)).astype(np.float32)
    
    u_blended = u_aligned * mask_2d + image * (1.0 - mask_2d)

    return np.clip(u_blended, 0, 1).astype(np.float32)

# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_nmse(reference, noisy):
    """Normalized Mean Square Error: ||ref - noisy||^2 / ||ref||^2."""
    mse = np.mean((reference - noisy) ** 2)
    norm_factor = np.mean(reference ** 2)
    return mse / norm_factor if norm_factor > 1e-12 else float("inf")


def compute_ssim(reference, noisy):
    """Structural Similarity Index Measure (SSIM) using local uniform windows."""
    # Constants based on standard SSIM formulation (dynamic range L = 1.0)
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    win_size = 7

    # Local means
    mu1 = ndimage.uniform_filter(reference, size=win_size)
    mu2 = ndimage.uniform_filter(noisy, size=win_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Local variances and covariance
    sigma1_sq = ndimage.uniform_filter(reference ** 2, size=win_size) - mu1_sq
    sigma2_sq = ndimage.uniform_filter(noisy ** 2, size=win_size) - mu2_sq
    sigma12 = ndimage.uniform_filter(reference * noisy, size=win_size) - mu1_mu2

    # SSIM map
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return np.mean(ssim_map)
