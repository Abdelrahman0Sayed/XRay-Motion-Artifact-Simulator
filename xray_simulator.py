#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   X-Ray Motion Artifact Simulator  ·  SBE 4220              ║
║   Medical Imaging II — Spring 2026                          ║
║                                                              ║
║   Physics implemented:                                       ║
║   • Beer-Lambert:  I = I₀ · exp(−∫ μ dl)                   ║
║   • Temporal integration:  P = (1/N) Σᵢ P(tᵢ)              ║
║   • Linear motion:    d(t) = v · t                          ║
║   • Sinusoidal motion: d(t) = A · sin(2π f t)               ║
║   • Poisson photon noise:  k ~ Poisson(N₀ · I)              ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, warnings, traceback
import numpy as np
from scipy import ndimage
from scipy.signal import wiener
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QSlider, QComboBox, QSpinBox, QPushButton,
    QStatusBar, QFrame, QProgressBar, QGridLayout, QSizePolicy,
    QScrollArea, QMessageBox,
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal
from PyQt5.QtGui   import QFont, QPalette, QColor


# ══════════════════════════════════════════════════════════════════
#  PHYSICS CONSTANTS
# ══════════════════════════════════════════════════════════════════

VOXEL_SIZE = 0.30   # cm per voxel  (phantom ~19.2 × 19.2 × 36 cm)

# Linear attenuation coefficients at 70 keV  (cm⁻¹)
# Source: NIST XCOM database / Hubbell & Seltzer 2004
MU = dict(
    air          = 0.000,
    fat          = 0.160,
    soft_tissue  = 0.200,
    muscle       = 0.210,
    blood        = 0.222,
    liver        = 0.215,
    lung         = 0.045,   # mostly air
    cortical     = 0.520,
    spongy_bone  = 0.370,
)

BODY_PARTS = {
    "Full Body" : (-1.00,  1.00),
    "Head"      : ( 0.58,  1.00),
    "Chest"     : ( 0.00,  0.57),
    "Abdomen"   : (-0.42,  0.05),
    "Pelvis"    : (-0.92, -0.38),
}

PROJ_AXES = {
    "AP  (Front → Back)"     : 1,
    "PA  (Back → Front)"     : 1,
    "Lateral  (Left → Right)": 0,
    "Lateral  (Right → Left)": 0,
}


# ══════════════════════════════════════════════════════════════════
#  PHANTOM GENERATION
# ══════════════════════════════════════════════════════════════════

def build_phantom(size=(64, 64, 120)):
    """
    Construct a 3-D numerical human phantom.

    Returns
    -------
    ndarray, shape (Nx, Ny, Nz), dtype float32
        Linear attenuation coefficients  μ  in cm⁻¹.
        Axes: x = left/right,  y = anterior/posterior,  z = inferior/superior.
    """
    Nx, Ny, Nz = size
    ph = np.zeros(size, dtype=np.float32)

    # Normalised coordinate grids  ∈ [-1, 1]
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    z = np.linspace(-1, 1, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    def ell(cx, cy, cz, rx, ry, rz):
        """Boolean mask of an axis-aligned ellipsoid."""
        return ((X-cx)/rx)**2 + ((Y-cy)/ry)**2 + ((Z-cz)/rz)**2 <= 1.0

    # ── Torso ────────────────────────────────────────────────────
    ph[ell(0, 0,  0.05, 0.38, 0.22, 0.50)] = MU["soft_tissue"]
    ph[ell(0, 0,  0.43, 0.46, 0.21, 0.09)] = MU["soft_tissue"]   # shoulders
    ph[ell(0, 0, -0.58, 0.36, 0.19, 0.10)] = MU["soft_tissue"]   # hip flare

    # ── Neck ─────────────────────────────────────────────────────
    ph[ell(0, 0, 0.60, 0.09, 0.08, 0.07)]  = MU["soft_tissue"]

    # ── Head ─────────────────────────────────────────────────────
    ph[ell(0, 0, 0.78, 0.20, 0.18, 0.19)]  = 0.205               # brain
    skull_o = ell(0, 0, 0.78, 0.216, 0.196, 0.206)
    skull_i = ell(0, 0, 0.78, 0.190, 0.170, 0.180)
    ph[skull_o & ~skull_i & (Z > 0.59)]     = MU["cortical"]

    # ── Spine (22 vertebrae) ──────────────────────────────────────
    for zv in np.linspace(-0.45, 0.52, 22):
        ph[ell(0.0, -0.10, zv, 0.042, 0.040, 0.022)] = MU["cortical"]
        ph[ell(0.0, -0.06, zv, 0.030, 0.028, 0.018)] = MU["spongy_bone"]

    # ── Ribs (10 pairs) — cosine arc approximation ───────────────
    for zv in np.linspace(0.05, 0.43, 10):
        for side in (-1, 1):
            for theta in np.linspace(0.05, np.pi - 0.05, 20):
                rx = side * 0.30 * np.sin(theta)
                ry = -0.08 + 0.17 * np.cos(theta)
                ph[ell(rx, ry, zv, 0.032, 0.028, 0.024)] = MU["cortical"]

    # ── Lungs ─────────────────────────────────────────────────────
    ph[ell(-0.17,  0.02, 0.22, 0.125, 0.090, 0.250)] = MU["lung"]
    ph[ell( 0.17,  0.02, 0.22, 0.125, 0.090, 0.250)] = MU["lung"]

    # ── Heart ─────────────────────────────────────────────────────
    ph[ell(-0.04, 0.01, 0.12, 0.090, 0.082, 0.112)]  = MU["blood"]

    # ── Liver ─────────────────────────────────────────────────────
    ph[ell(0.13, 0.05, -0.08, 0.185, 0.125, 0.105)]  = MU["liver"]

    # ── Stomach / soft abdominal organs ───────────────────────────
    ph[ell(-0.10, 0.04, -0.12, 0.10, 0.09, 0.08)]    = MU["soft_tissue"]
    ph[ell( 0.05, 0.03, -0.22, 0.13, 0.10, 0.09)]    = MU["soft_tissue"]

    # ── Pelvis ────────────────────────────────────────────────────
    ph[ell(0, 0, -0.62, 0.35, 0.19, 0.12)]            = MU["soft_tissue"]
    pbo = ell(0, 0, -0.62, 0.375, 0.215, 0.138)
    pbi = ell(0, 0, -0.62, 0.305, 0.160, 0.102)
    ph[pbo & ~pbi]                                     = MU["spongy_bone"]

    # ── Arms ──────────────────────────────────────────────────────
    for side in (-1, 1):
        ax = side * 0.60
        ph[ell(ax, 0,  0.18, 0.072, 0.062, 0.285)] = MU["soft_tissue"]
        ph[ell(ax, 0,  0.18, 0.026, 0.024, 0.265)] = MU["cortical"]    # humerus
        ph[ell(ax, 0, -0.38, 0.063, 0.055, 0.195)] = MU["soft_tissue"] # forearm

    # ── Upper legs ────────────────────────────────────────────────
    for side in (-1, 1):
        lx = side * 0.22
        ph[ell(lx, 0, -0.82, 0.115, 0.095, 0.145)] = MU["soft_tissue"]
        ph[ell(lx, 0, -0.82, 0.036, 0.032, 0.125)] = MU["cortical"]    # femur

    return ph


def crop_phantom(phantom, body_part):
    """Return sub-volume corresponding to the chosen body part."""
    z_range = BODY_PARTS.get(body_part, (-1.0, 1.0))
    Nz = phantom.shape[2]
    z_lin = np.linspace(-1, 1, Nz)
    mask  = (z_lin >= z_range[0]) & (z_lin <= z_range[1])
    return phantom[:, :, mask]


# ══════════════════════════════════════════════════════════════════
#  X-RAY PHYSICS
# ══════════════════════════════════════════════════════════════════

def project_parallel(volume, axis):
    """
    Parallel-beam projection using Beer-Lambert law.

        Line integral :  L(u,v) = Σᵢ  μᵢ(u,v) · Δl      [cm⁻¹ · cm = dimensionless]
        Transmitted   :  I(u,v) = I₀ · exp(−L(u,v))

    Parameters
    ----------
    volume : ndarray (Nx, Ny, Nz)  — attenuation coefficients in cm⁻¹
    axis   : int                   — projection axis (0=x, 1=y)

    Returns
    -------
    ndarray (2-D)  — normalised intensity  ∈ (0, 1]
    """
    L = np.sum(volume, axis=axis, dtype=np.float64) * VOXEL_SIZE   # cm
    return np.exp(-L).astype(np.float32)


def simulate_acquisition(volume, p):
    """
    Temporal integration of projections over the exposure window T.

        P_final(u,v) = (1/T) ∫₀ᵀ  P(u,v; t)  dt
                     ≈ (1/N) Σᵢ₌₀ᴺ⁻¹  P(u,v; tᵢ)          [midpoint rule]

    Motion models
    -------------
    Linear      :  d(t) = v · t                             [cm]
    Breathing   :  d(t) = A · sin(2π f t)                   [cm]
    Cardiac     :  d(t) = A · sin(2π f t)   (higher f, multi-axis)
    """
    mtype = p["motion_type"]

    # Fast path — no motion
    if mtype == "none":
        return project_parallel(volume, p["proj_axis"])

    T      = p["exposure_time"]    # s
    N      = p["n_steps"]
    v      = p["velocity"]         # cm/s  (linear)
    A      = p["amplitude"]        # cm    (sinusoidal)
    f      = p["frequency"]        # Hz
    axis   = p["proj_axis"]
    maxis  = p["motion_axis"]      # 0 = x,  2 = z

    accum = None
    dt    = T / N

    for i in range(N):
        t = (i + 0.5) * dt          # midpoint of sub-interval

        # ── displacement  d(t)  ──────────────────────────────────
        if   mtype == "linear":
            d_cm = v * t
        elif mtype == "breathing":
            d_cm = A * np.sin(2 * np.pi * f * t)
        elif mtype == "cardiac":
            d_cm = A * np.sin(2 * np.pi * f * t)
        else:
            d_cm = 0.0

        d_vox = d_cm / VOXEL_SIZE   # convert cm → voxels

        # ── apply rigid-body shift to phantom ────────────────────
        shift = [0.0, 0.0, 0.0]
        shift[maxis] = d_vox
        if abs(d_vox) > 0.01:
            moved = ndimage.shift(volume, shift, mode="nearest", order=1)
        else:
            moved = volume

        # ── Beer-Lambert projection ───────────────────────────────
        proj  = project_parallel(moved, axis)
        accum = proj if accum is None else accum + proj

    return (accum / N).astype(np.float32)


def add_noise(proj, noise_type, N0):
    """
    Detector noise models.

    Poisson (quantum / photon-counting noise):
        Detected quanta :  k ~ Poisson(N₀ · I)
        Normalised image:  Î = k / N₀
        Quantum SNR     ≈  √(N₀ · Ī)

    Gaussian (electronic read-out noise):
        σ_read = 1 / √N₀

    Combined (Poisson shot noise + Gaussian read noise):
        realistic flat-panel detector model
    """
    p = proj.copy()
    p = np.clip(p, 0.0, 1.0)

    if   noise_type == "None":
        return p

    elif noise_type == "Poisson":
        counts = np.random.poisson(N0 * p)
        return np.clip(counts / N0, 0, 1).astype(np.float32)

    elif noise_type == "Gaussian":
        σ = 1.0 / np.sqrt(max(N0, 1))
        return np.clip(p + np.random.normal(0, σ, p.shape), 0, 1).astype(np.float32)

    elif noise_type == "Combined":
        counts  = np.random.poisson(N0 * p)
        σ_read  = np.sqrt(N0) * 0.01        # ~1 % read noise relative to full scale
        read    = np.random.normal(0, σ_read, p.shape)
        return np.clip((counts + read) / N0, 0, 1).astype(np.float32)

    return p


def apply_mitigation(image, method):
    """Post-acquisition artifact mitigation / image-restoration methods."""
    img = image.copy()

    if   method == "None":
        return img
    elif method == "Median Filter":
        return ndimage.median_filter(img, size=3).astype(np.float32)
    elif method == "Gaussian Smooth":
        return ndimage.gaussian_filter(img, sigma=0.9).astype(np.float32)
    elif method == "Wiener Filter":
        w = wiener(img, mysize=7)
        return np.clip(w, 0, 1).astype(np.float32)
    elif method == "Unsharp Mask":
        blur = ndimage.gaussian_filter(img, sigma=1.5)
        return np.clip(img + 0.65 * (img - blur), 0, 1).astype(np.float32)
    elif method == "RL Deconvolution":
        return _rl_deconv(img, psf=9, iters=12)
    return img


def _rl_deconv(image, psf=9, iters=12):
    """
    Richardson-Lucy deconvolution.
    PSF approximated as a uniform box kernel (models linear motion blur).

        u^(k+1) = u^(k) · ( h̃ * (d / (h * u^(k))) )

    where  h  is the PSF,  h̃ = flip(h),  d  is the observed image.
    """
    psf_k = np.ones((psf, psf), dtype=np.float32) / (psf * psf)
    u = image.copy()
    for _ in range(iters):
        conv = ndimage.convolve(u, psf_k, mode="reflect")
        conv = np.where(conv < 1e-12, 1e-12, conv)
        ratio = image / conv
        u = u * ndimage.convolve(ratio, psf_k[::-1, ::-1], mode="reflect")
    return np.clip(u, 0, 1).astype(np.float32)


def compute_snr(reference, noisy):
    """
    Signal-to-Noise Ratio:
        SNR = 20 · log₁₀( RMS_signal / RMS_noise )   [dB]
    """
    noise  = noisy - reference
    rms_s  = np.sqrt(np.mean(reference ** 2))
    rms_n  = np.sqrt(np.mean(noise    ** 2))
    return 20 * np.log10(rms_s / rms_n) if rms_n > 1e-12 else float("inf")


def compute_psnr(reference, noisy):
    """Peak SNR in dB."""
    mse = np.mean((reference - noisy) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 1e-12 else float("inf")


# ══════════════════════════════════════════════════════════════════
#  MATPLOTLIB CANVAS  —  3-D PHANTOM VIEWER
# ══════════════════════════════════════════════════════════════════

class Phantom3DCanvas(FigureCanvas):
    """Interactive 3-D point-cloud rendering of the digital phantom."""

    BG = "#0b0b18"

    def __init__(self, phantom, parent=None):
        self.fig = Figure(figsize=(5, 7.2), facecolor=self.BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.phantom = phantom
        self._tube_artists = []

        self._ax = self.fig.add_subplot(111, projection="3d")
        self._ax.set_facecolor(self.BG)
        self.fig.patch.set_facecolor(self.BG)

        self._build_point_cloud()
        self._draw_base()

    # ── point cloud ──────────────────────────────────────────────
    def _build_point_cloud(self):
        ph  = self.phantom
        Nx, Ny, Nz = ph.shape
        xi, yi, zi = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz),
                                  indexing="ij")
        coords = np.stack([xi.ravel(), yi.ravel(), zi.ravel()], axis=1)
        vals   = ph.ravel()
        rng    = np.random.default_rng(7)

        pts, cols = [], []

        def sample(mask_flat, color, frac):
            idx = np.where(mask_flat)[0]
            if len(idx) == 0:
                return
            n = max(1, int(len(idx) * frac))
            chosen = rng.choice(idx, min(n, len(idx)), replace=False)
            pts.append(coords[chosen])
            cols.extend([color] * len(chosen))

        bone_m  = vals > 0.45
        lung_m  = (vals > 0.01) & (vals < 0.10)
        organ_m = (vals > 0.21) & (vals < 0.40)

        # Soft tissue — surface voxels only (avoid interior mass)
        soft_all  = ((vals > 0.13) & (vals < 0.45)).reshape(ph.shape)
        soft_surf = soft_all & ~ndimage.binary_erosion(soft_all)
        soft_flat = soft_surf.ravel()

        sample(bone_m,  "#e8e8e8", 0.22)
        sample(soft_flat, "#b07850", 0.14)
        sample(lung_m,  "#6aadcc", 0.14)
        sample(organ_m & ~soft_flat, "#bb3333", 0.10)

        self._pc = np.vstack(pts) if pts else np.zeros((1, 3))
        self._pc_colors = cols if cols else ["#666666"]

    def _draw_base(self):
        ax = self._ax
        ax.cla()
        ax.set_facecolor(self.BG)

        c = self._pc
        # Plot: x-axis, z-axis (superior = vertical), y-axis (depth)
        ax.scatter(c[:, 0], c[:, 2], c[:, 1],
                   c=self._pc_colors, s=1.2, alpha=0.65,
                   depthshade=True, rasterized=True)

        ax.set_xlabel("X", color="#5566aa", fontsize=7, labelpad=2)
        ax.set_ylabel("Z  (Sup→)", color="#5566aa", fontsize=7, labelpad=2)
        ax.set_zlabel("Y  (Depth)", color="#5566aa", fontsize=7, labelpad=2)
        ax.tick_params(colors="#334466", labelsize=5)
        ax.set_title("3-D Phantom", color="#8899cc", fontsize=9, pad=3)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#1a1a38")
        ax.view_init(elev=15, azim=-65)
        self.draw_idle()

    def update_tube(self, body_part, proj_direction):
        """Overlay X-ray source ☀ and detector plane on the 3-D view."""
        ax = self._ax

        for a in self._tube_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._tube_artists.clear()

        Nx, Ny, Nz = self.phantom.shape
        cx, cy = Nx // 2, Ny // 2

        z_norm = BODY_PARTS.get(body_part, (-1.0, 1.0))
        z_lo = int((z_norm[0] + 1) / 2 * Nz)
        z_hi = int((z_norm[1] + 1) / 2 * Nz)
        z_ctr = (z_lo + z_hi) // 2
        dz    = max((z_hi - z_lo) // 3, 4)

        axis_idx = PROJ_AXES.get(proj_direction, 1)
        BEAM_COLOR = "#ffe044"
        SRC_COLOR  = "#ffcc00"
        DET_COLOR  = "#44eebb"

        if axis_idx == 1:                       # AP / PA
            y_src, y_det = -14, Ny + 14
            for dxi in (-Nx//5, Nx//5):
                for dzi in (-dz, dz):
                    ln, = ax.plot([cx+dxi, cx], [z_ctr+dzi, z_ctr], [y_src, cy],
                                  color=BEAM_COLOR, alpha=0.28, lw=0.8, linestyle="--")
                    self._tube_artists.append(ln)
            sc = ax.scatter([cx], [z_ctr], [y_src], c=SRC_COLOR, s=90,
                            marker="*", zorder=8, depthshade=False)
            dc = ax.scatter([cx], [z_ctr], [y_det], c=DET_COLOR, s=45,
                            marker="s", zorder=8, depthshade=False)
        else:                                   # Lateral
            x_src, x_det = -14, Nx + 14
            for dzi in (-dz, dz):
                ln, = ax.plot([x_src, cx], [z_ctr+dzi, z_ctr], [cy, cy],
                              color=BEAM_COLOR, alpha=0.28, lw=0.8, linestyle="--")
                self._tube_artists.append(ln)
            sc = ax.scatter([x_src], [z_ctr], [cy], c=SRC_COLOR, s=90,
                            marker="*", zorder=8, depthshade=False)
            dc = ax.scatter([x_det], [z_ctr], [cy], c=DET_COLOR, s=45,
                            marker="s", zorder=8, depthshade=False)

        # Target region highlight
        tgt = ax.scatter([cx], [z_ctr], [cy],
                         c="#ff3322", s=60, marker="o",
                         alpha=0.80, zorder=9, depthshade=False)
        self._tube_artists += [sc, dc, tgt]

        # Region box
        for zv in (z_lo, z_hi):
            bx, = ax.plot([0, Nx, Nx, 0, 0], [zv, zv, zv, zv, zv],
                           [0, 0, Ny, Ny, 0], color="#ff3322",
                           alpha=0.18, lw=0.7)
            self._tube_artists.append(bx)

        self.draw_idle()


# ══════════════════════════════════════════════════════════════════
#  MATPLOTLIB CANVAS  —  2-D PROJECTION RESULTS
# ══════════════════════════════════════════════════════════════════

class Projection2DCanvas(FigureCanvas):
    """Three-panel display:  static  |  motion blur  |  mitigated."""

    BG = "#090912"

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 4.5), facecolor=self.BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fig.patch.set_facecolor(self.BG)
        self.film_mode = True
        self.show_centerlines = False
        self._last_results = None
        self._placeholder()

    def set_film_mode(self, enabled):
        self.film_mode = bool(enabled)
        if self._last_results is not None:
            self.show_results(*self._last_results)

    def set_centerlines_enabled(self, enabled):
        self.show_centerlines = bool(enabled)
        if self._last_results is not None:
            self.show_results(*self._last_results)

    def _placeholder(self):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#0d0d1e")
        ax.text(0.5, 0.52,
                "No exposure yet.",
                ha="center", va="center",
                color="#3a4a6a", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.42,
                "Configure parameters and press  ☢  SHOOT X-RAY",
                ha="center", va="center",
                color="#2a3a58", fontsize=9, transform=ax.transAxes)
        ax.axis("off")
        self.draw_idle()

    def show_results(self, static, motion, mitigated, params):
        self._last_results = (static, motion, mitigated, params)
        self.fig.clf()
        cmap = "gray"
        titles = [
            "① Static  (no motion)",
            "② Motion Artifact\n" + _motion_label(params),
            f"③ Mitigated\n({params['mitigation']})",
        ]
        images = [static, motion, mitigated]
        colors = ["#66aaff", "#ff6644", "#44ee88"]

        for k, (img, ttl, col) in enumerate(zip(images, titles, colors)):
            ax = self.fig.add_subplot(1, 3, k + 1)
            ax.set_facecolor("#000008")
            display_img = (1.0 - img) if self.film_mode else img
            ax.imshow(display_img.T, cmap=cmap, vmin=0, vmax=1,
                      aspect="auto", interpolation="bilinear",
                      origin="lower")
            if self.show_centerlines:
                h, w = display_img.T.shape
                x_mid = (w - 1) / 2.0
                y_mid = (h - 1) / 2.0
                ax.axvline(x_mid, color="#ffde59", linestyle="--",
                           linewidth=0.8, alpha=0.9)
                ax.axhline(y_mid, color="#ffde59", linestyle="--",
                           linewidth=0.8, alpha=0.9)
            ax.set_title(ttl, color=col, fontsize=7.5,
                         pad=5, fontweight="bold")
            ax.axis("off")
            # Border colour matching title
            for spine in ax.spines.values():
                spine.set_edgecolor(col)
                spine.set_linewidth(1.2)
                spine.set_visible(True)

        self.fig.suptitle(
            f"Exposure {params['exposure_time']:.2f} s  ·  "
            f"N₀ = {params['n_photons']:,}",
            color="#5566aa", fontsize=7.5, y=0.01
        )
        self.fig.tight_layout(rect=[0, 0.05, 1, 0.97])
        self.draw_idle()


def _motion_label(p):
    mt = p["motion_type"]
    if mt == "none":
        return "(no motion)"
    elif mt == "linear":
        return f"linear  v={p['velocity']:.1f} cm/s"
    elif mt == "breathing":
        return f"breathing  A={p['amplitude']:.1f} cm  f={p['frequency']:.2f} Hz"
    elif mt == "cardiac":
        return f"cardiac  A={p['amplitude']:.1f} cm  f={p['frequency']:.2f} Hz"
    return mt


# ══════════════════════════════════════════════════════════════════
#  BACKGROUND SIMULATION THREAD
# ══════════════════════════════════════════════════════════════════

class SimThread(QThread):
    done     = pyqtSignal(object, object, object, dict)
    progress = pyqtSignal(int)
    error    = pyqtSignal(str)

    def __init__(self, phantom, params):
        super().__init__()
        self.phantom = phantom
        self.params  = params

    def run(self):
        try:
            ph = self.phantom
            p  = self.params

            self.progress.emit(10)
            vol = crop_phantom(ph, p["body_part"])

            self.progress.emit(20)
            static_proj = project_parallel(vol, p["proj_axis"])

            self.progress.emit(30)
            motion_proj = simulate_acquisition(vol, p)

            self.progress.emit(72)
            static_noisy = add_noise(static_proj, p["noise_type"], p["n_photons"])
            noisy        = add_noise(motion_proj, p["noise_type"], p["n_photons"])

            self.progress.emit(83)
            mitigated = apply_mitigation(noisy, p["mitigation"])

            self.progress.emit(95)
            snr_motion = compute_snr(static_proj, noisy)
            snr_mitig  = compute_snr(static_proj, mitigated)
            psnr_m     = compute_psnr(static_proj, noisy)
            psnr_r     = compute_psnr(static_proj, mitigated)

            result_p = {**p,
                        "snr_motion":    snr_motion,
                        "snr_mitigated": snr_mitig,
                        "psnr_motion":   psnr_m,
                        "psnr_mitig":    psnr_r}
            self.progress.emit(100)
            self.done.emit(static_noisy, noisy, mitigated, result_p)

        except Exception:
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════
#  STYLING
# ══════════════════════════════════════════════════════════════════

APP_STYLE = """
QMainWindow, QWidget {
    background-color: #0e0e1c;
    color: #c8cce8;
    font-family: "Segoe UI", "SF Pro Display", Arial, sans-serif;
    font-size: 12px;
}
QScrollArea, QScrollArea > QWidget > QWidget {
    background-color: #0e0e1c;
    border: none;
}
QGroupBox {
    border: 1px solid #1e2a4a;
    border-radius: 7px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
    font-size: 11px;
    color: #6680bb;
    letter-spacing: 0.5px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    background-color: #0e0e1c;
}
QComboBox, QSpinBox, QDoubleSpinBox {
    background: #161628;
    border: 1px solid #2a3560;
    border-radius: 5px;
    color: #c8cce8;
    padding: 4px 8px;
    min-height: 26px;
    selection-background-color: #2a3560;
}
QComboBox:hover, QSpinBox:hover { border-color: #4466aa; }
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    background: #0f1020;
    border: 1px solid #1f2744;
    color: #566084;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow { width: 10px; height: 10px; }
QComboBox QAbstractItemView {
    background: #161628;
    color: #c8cce8;
    selection-background-color: #2a4080;
    border: 1px solid #2a3560;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #1e2a4a;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #3d66cc;
    width: 14px; height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    border: 2px solid #5588ff;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #2244aa, stop:1 #4466ee);
    border-radius: 2px;
}
QSlider:disabled::groove:horizontal { background: #14182a; }
QSlider:disabled::handle { background: #333344; border-color: #333344; }
QSlider:disabled::sub-page { background: #1e2030; }
QPushButton#shoot_btn {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #cc2200, stop:1 #881500);
    color: #ffddcc;
    font-size: 15px;
    font-weight: 700;
    border-radius: 9px;
    padding: 12px 24px;
    border: 2px solid #ff4422;
    letter-spacing: 3px;
    min-height: 52px;
}
QPushButton#shoot_btn:hover {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #ee3300, stop:1 #aa2200);
    border-color: #ff6644;
}
QPushButton#shoot_btn:pressed {
    background: #881100;
}
QPushButton#shoot_btn:disabled {
    background: #1e2030;
    border-color: #2a2a44;
    color: #444466;
}
QLabel { color: #c8cce8; }
QLabel#val_lbl { color: #5599ff; font-weight: 700; font-size: 11px; }
QLabel#val_lbl:disabled { color: #4f5b82; }
QLabel#header  { color: #7799dd; font-size: 11px; font-weight: 600; }
QLabel#disabled_hint { color: #6e7aa3; font-size: 10px; font-style: italic; }
QPushButton#toggle_btn {
    background: #182441;
    border: 1px solid #30508a;
    border-radius: 6px;
    color: #a9c4ff;
    font-size: 10px;
    padding: 4px 8px;
    min-height: 22px;
}
QPushButton#toggle_btn:checked {
    background: #254f95;
    border-color: #4b8dff;
    color: #eaf2ff;
}
QPushButton#toggle_btn:hover { border-color: #5f9fff; }
QProgressBar {
    border: 1px solid #1e2a4a;
    border-radius: 4px;
    background: #111122;
    text-align: center;
    color: #8899cc;
    max-height: 14px;
    font-size: 10px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #2244aa, stop:1 #44aaff);
    border-radius: 3px;
}
QStatusBar { color: #4a5a7a; font-size: 11px; }
QFrame#divider { background: #1a2040; max-height: 1px; }
"""


# ══════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("  ☢   X-Ray Motion Artifact Simulator  ·  SBE 4220")
        self.setMinimumSize(1380, 780)
        self.setStyleSheet(APP_STYLE)

        # Internal state
        self._sim_thread = None
        self._exp_val  = 0.50
        self._vel_val  = 1.50
        self._amp_val  = 1.50
        self._freq_val = 0.30

        # Build phantom (once)
        self.statusBar().showMessage("Building 3-D phantom …  please wait")
        QApplication.processEvents()
        self.phantom = build_phantom((64, 64, 120))

        self._build_ui()
        self._on_motion_type_changed(self.cb_motion.currentText())
        self._refresh_tube()
        self.statusBar().showMessage(
            "Phantom ready.  Adjust parameters and press  ☢  SHOOT X-RAY")

    # ── UI Construction ──────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        # ── LEFT: 3-D Phantom ─────────────────────────────────────
        left_box = QFrame()
        left_box.setStyleSheet(
            "QFrame { border: 1px solid #1a2648; border-radius:8px; }")
        left_lay = QVBoxLayout(left_box)
        left_lay.setContentsMargins(3, 3, 3, 3)

        lbl_3d = QLabel("  3-D PHANTOM VIEWER")
        lbl_3d.setObjectName("header")
        lbl_3d.setStyleSheet(
            "background:#0f1428; color:#4466aa; padding:6px; "
            "font-size:10px; letter-spacing:1px; border-radius:4px;")
        left_lay.addWidget(lbl_3d)

        self.canvas3d = Phantom3DCanvas(self.phantom)
        left_lay.addWidget(self.canvas3d)

        # Legend
        leg_row = QHBoxLayout()
        for txt, col in [("● Bone", "#e8e8e8"), ("● Tissue", "#b07850"),
                          ("● Lung", "#6aadcc"),  ("● Organ", "#bb3333"),
                          ("★ Source", "#ffcc00"), ("■ Detector", "#44eebb")]:
            lb = QLabel(txt)
            lb.setStyleSheet(f"color:{col}; font-size:9px;")
            leg_row.addWidget(lb)
        left_lay.addLayout(leg_row)

        root.addWidget(left_box, stretch=38)

        # ── CENTER: Controls ──────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(265)
        scroll.setMaximumWidth(295)
        ctrl_w = QWidget()
        ctrl_lay = QVBoxLayout(ctrl_w)
        ctrl_lay.setSpacing(8)
        ctrl_lay.setContentsMargins(4, 4, 4, 4)

        ctrl_lay.addWidget(self._grp_target())
        ctrl_lay.addWidget(self._grp_xray())
        ctrl_lay.addWidget(self._grp_motion())
        ctrl_lay.addWidget(self._grp_mitigation())
        ctrl_lay.addWidget(self._grp_view())

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        ctrl_lay.addWidget(self.progress)

        self.shoot_btn = QPushButton("☢   SHOOT X-RAY")
        self.shoot_btn.setObjectName("shoot_btn")
        self.shoot_btn.clicked.connect(self._on_shoot)
        ctrl_lay.addWidget(self.shoot_btn)

        ctrl_lay.addStretch(1)
        scroll.setWidget(ctrl_w)
        root.addWidget(scroll, stretch=22)

        # ── RIGHT: 2-D Projections ────────────────────────────────
        right_box = QFrame()
        right_box.setStyleSheet(
            "QFrame { border: 1px solid #1a2648; border-radius:8px; }")
        right_lay = QVBoxLayout(right_box)
        right_lay.setContentsMargins(3, 3, 3, 3)

        lbl_2d = QLabel("  2-D PROJECTION RESULTS")
        lbl_2d.setStyleSheet(
            "background:#0f1428; color:#4466aa; padding:6px; "
            "font-size:10px; letter-spacing:1px; border-radius:4px;")
        right_lay.addWidget(lbl_2d)

        self.canvas2d = Projection2DCanvas()
        right_lay.addWidget(self.canvas2d)

        # Metrics row
        metrics = QHBoxLayout()
        self.lbl_snr_m  = QLabel("Motion SNR: —")
        self.lbl_snr_r  = QLabel("Mitigated SNR: —")
        self.lbl_psnr_m = QLabel("PSNR: —")
        for lb in (self.lbl_snr_m, self.lbl_snr_r, self.lbl_psnr_m):
            lb.setStyleSheet("color:#4488dd; font-size:10px; font-weight:600;")
            metrics.addWidget(lb)
        right_lay.addLayout(metrics)

        root.addWidget(right_box, stretch=48)

    # ── Control Groups ────────────────────────────────────────────

    def _grp_target(self):
        g = QGroupBox("🎯  TARGET  &  DIRECTION")
        lay = QGridLayout(g)
        lay.setSpacing(5)

        lay.addWidget(QLabel("Body Part:"),     0, 0)
        self.cb_part = QComboBox()
        self.cb_part.addItems(list(BODY_PARTS.keys()))
        self.cb_part.setCurrentText("Chest")
        self.cb_part.currentTextChanged.connect(self._refresh_tube)
        lay.addWidget(self.cb_part, 0, 1)

        lay.addWidget(QLabel("Projection:"),    1, 0)
        self.cb_proj = QComboBox()
        self.cb_proj.addItems(list(PROJ_AXES.keys()))
        self.cb_proj.currentTextChanged.connect(self._refresh_tube)
        lay.addWidget(self.cb_proj, 1, 1)

        lay.addWidget(QLabel("Motion Axis:"),   2, 0)
        self.cb_maxis = QComboBox()
        self.cb_maxis.addItems(["X  (Left–Right)", "Z  (Head–Foot)"])
        lay.addWidget(self.cb_maxis, 2, 1)

        return g

    def _slider_row(self, label, lo, hi, init, dec, unit, attr):
        """Build a compact slider row; store reference in self.<attr>."""
        w   = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)

        lb = QLabel(label)
        lb.setFixedWidth(88)
        base_label = label.rstrip()
        if not base_label.endswith(":"):
            base_label += ":"
        lb.setProperty("base_text", base_label)
        row.addWidget(lb)

        sl = QSlider(Qt.Horizontal)
        sl.setRange(0, 1000)
        sl.setValue(int((init - lo) / (hi - lo) * 1000))

        val_lb = QLabel(f"{init:.{dec}f} {unit}")
        val_lb.setObjectName("val_lbl")
        val_lb.setFixedWidth(68)

        def _on_change(v, _lo=lo, _hi=hi, _dec=dec, _unit=unit, _attr=attr, _lb=val_lb):
            real = _lo + v / 1000 * (_hi - _lo)
            _lb.setText(f"{real:.{_dec}f} {_unit}")
            setattr(self, _attr, real)

        sl.valueChanged.connect(_on_change)
        row.addWidget(sl)
        row.addWidget(val_lb)

        setattr(self, f"_sl_{attr}", sl)
        setattr(self, f"_lb_{attr}", lb)
        return w

    def _grp_xray(self):
        g = QGroupBox("⚡  X-RAY  PARAMETERS")
        lay = QVBoxLayout(g)
        lay.setSpacing(5)
        lay.addWidget(self._slider_row(
            "Exposure (s):", 0.01, 3.0, 0.5, 2, "s", "_exp_val"))

        row = QHBoxLayout()
        row.addWidget(QLabel("Photon Flux N₀:"))
        self.cb_flux = QComboBox()
        for n, lbl in [(500, "500  (extreme low)"),
                        (2000,  "2 k  (very low)"),
                        (10000, "10 k  (low)"),
                        (50000, "50 k  (medium)"),
                        (200000,"200 k  (high)"),
                        (1000000,"1 M  (very high)")]:
            self.cb_flux.addItem(lbl, n)
        self.cb_flux.setCurrentIndex(3)
        row.addWidget(self.cb_flux)
        lay.addLayout(row)
        return g

    def _grp_motion(self):
        g = QGroupBox("🏃  MOTION  PARAMETERS")
        lay = QVBoxLayout(g)
        lay.setSpacing(5)

        row = QHBoxLayout()
        row.addWidget(QLabel("Motion Type:"))
        self.cb_motion = QComboBox()
        self.cb_motion.addItems(["none", "linear", "breathing", "cardiac"])
        self.cb_motion.setCurrentText("breathing")
        self.cb_motion.currentTextChanged.connect(self._on_motion_type_changed)
        row.addWidget(self.cb_motion)
        lay.addLayout(row)

        self._w_vel  = self._slider_row(
            "Velocity:",  0.0, 15.0, 1.5, 2, "cm/s", "_vel_val")
        self._w_amp  = self._slider_row(
            "Amplitude:", 0.0,  5.0, 1.5, 2, "cm",   "_amp_val")
        self._w_freq = self._slider_row(
            "Frequency:", 0.1,  3.0, 0.3, 2, "Hz",   "_freq_val")
        lay.addWidget(self._w_vel)
        lay.addWidget(self._w_amp)
        lay.addWidget(self._w_freq)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Integration Steps:"))
        self.sb_steps = QSpinBox()
        self.sb_steps.setRange(5, 200)
        self.sb_steps.setValue(40)
        self.sb_steps.setSuffix("  steps")
        row2.addWidget(self.sb_steps)
        lay.addLayout(row2)

        self.lbl_motion_hint = QLabel("")
        self.lbl_motion_hint.setObjectName("disabled_hint")
        self.lbl_motion_hint.setWordWrap(True)
        lay.addWidget(self.lbl_motion_hint)

        return g

    def _grp_mitigation(self):
        g = QGroupBox("🔧  MITIGATION  STRATEGY")
        lay = QHBoxLayout(g)
        lay.addWidget(QLabel("Method:"))
        self.cb_mitig = QComboBox()
        self.cb_mitig.addItems([
            "None",
            "Median Filter",
            "Gaussian Smooth",
            "Wiener Filter",
            "Unsharp Mask",
            "RL Deconvolution",
        ])
        self.cb_mitig.setCurrentText("Wiener Filter")
        lay.addWidget(self.cb_mitig)
        return g

    def _grp_view(self):
        g = QGroupBox("🖼️  RESULT  VIEW")
        lay = QVBoxLayout(g)
        lay.setSpacing(5)

        self.btn_film_mode = QPushButton("Film Contrast (Absorption Bright)")
        self.btn_film_mode.setObjectName("toggle_btn")
        self.btn_film_mode.setCheckable(True)
        self.btn_film_mode.setChecked(True)
        self.btn_film_mode.toggled.connect(self._on_view_option_changed)
        lay.addWidget(self.btn_film_mode)

        self.btn_centerlines = QPushButton("Show Centerline Axes")
        self.btn_centerlines.setObjectName("toggle_btn")
        self.btn_centerlines.setCheckable(True)
        self.btn_centerlines.setChecked(False)
        self.btn_centerlines.toggled.connect(self._on_view_option_changed)
        lay.addWidget(self.btn_centerlines)

        return g

    # ── Slots ─────────────────────────────────────────────────────

    def _on_motion_type_changed(self, mtype):
        lin = mtype == "linear"
        sin = mtype in ("breathing", "cardiac")
        none_m = mtype == "none"

        self._set_slider_row_enabled(self._w_vel, self._lb__vel_val, lin)
        self._set_slider_row_enabled(self._w_amp, self._lb__amp_val, sin)
        self._set_slider_row_enabled(self._w_freq, self._lb__freq_val, sin)

        self.cb_maxis.setEnabled(lin)
        self.cb_maxis.setToolTip("" if lin else "Motion axis is only used in linear mode")
        self.sb_steps.setEnabled(not none_m)

        if none_m:
            self.lbl_motion_hint.setText("Motion controls are disabled in none mode.")
        elif lin:
            self.lbl_motion_hint.setText("Amplitude/Frequency are disabled in linear mode.")
        else:
            self.lbl_motion_hint.setText("Velocity and direction are disabled in breathing/cardiac.")

    def _set_slider_row_enabled(self, row_widget, label_widget, enabled):
        row_widget.setEnabled(enabled)
        row_widget.setToolTip("" if enabled else "Disabled for selected motion type")
        base = label_widget.property("base_text") or label_widget.text()
        label_widget.setText(base if enabled else f"{base} (disabled)")

    def _on_view_option_changed(self):
        self.canvas2d.set_film_mode(self.btn_film_mode.isChecked())
        self.canvas2d.set_centerlines_enabled(self.btn_centerlines.isChecked())
        if getattr(self.canvas2d, "figure", None) and self.canvas2d.figure.axes:
            self.canvas2d.draw_idle()

    def _refresh_tube(self):
        if hasattr(self, "canvas3d"):
            self.canvas3d.update_tube(
                self.cb_part.currentText(),
                self.cb_proj.currentText())

    def _collect_params(self):
        mtype  = self.cb_motion.currentText()
        maxis  = 0 if "X" in self.cb_maxis.currentText() else 2
        if mtype != "linear":
            maxis = 2
        return {
            "body_part"    : self.cb_part.currentText(),
            "proj_axis"    : PROJ_AXES[self.cb_proj.currentText()],
            "exposure_time": self._exp_val,
            "n_photons"    : self.cb_flux.currentData(),
            "motion_type"  : mtype,
            "velocity"     : self._vel_val  if mtype == "linear" else 0.0,
            "amplitude"    : self._amp_val,
            "frequency"    : self._freq_val,
            "motion_axis"  : maxis,
            "n_steps"      : self.sb_steps.value(),
            "noise_type"   : "None",
            "mitigation"   : self.cb_mitig.currentText(),
        }

    def _on_shoot(self):
        if self._sim_thread and self._sim_thread.isRunning():
            return
        params = self._collect_params()
        self.shoot_btn.setEnabled(False)
        self.shoot_btn.setText("☢   Simulating …")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.statusBar().showMessage(
            f"Simulating … {params['motion_type']} motion | "
            f"{params['n_steps']} integration steps | "
            "noise disabled")

        self._sim_thread = SimThread(self.phantom, params)
        self._sim_thread.done.connect(self._on_sim_done)
        self._sim_thread.progress.connect(self.progress.setValue)
        self._sim_thread.error.connect(self._on_sim_error)
        self._sim_thread.start()

    def _on_sim_done(self, static, motion, mitigated, p):
        self._on_view_option_changed()
        self.canvas2d.show_results(static, motion, mitigated, p)

        snr_m = p.get("snr_motion",    float("nan"))
        snr_r = p.get("snr_mitigated", float("nan"))
        psnr  = p.get("psnr_mitig",    float("nan"))

        def _fmt_db(v):
            return f"{v:.1f} dB" if np.isfinite(v) else "—"

        self.lbl_snr_m .setText(f"Motion SNR: {_fmt_db(snr_m)}")
        self.lbl_snr_r .setText(f"Mitigated SNR: {_fmt_db(snr_r)}")
        self.lbl_psnr_m.setText(f"PSNR: {_fmt_db(psnr)}")

        self.shoot_btn.setEnabled(True)
        self.shoot_btn.setText("☢   SHOOT X-RAY")
        self.progress.setVisible(False)

        delta = snr_r - snr_m if np.isfinite(snr_m) and np.isfinite(snr_r) else 0
        arrow = "▲" if delta >= 0 else "▼"
        self.statusBar().showMessage(
            f"Done.   Motion SNR = {_fmt_db(snr_m)}  →  "
            f"After mitigation = {_fmt_db(snr_r)}  ({arrow}{abs(delta):.1f} dB)")

    def _on_sim_error(self, msg):
        QMessageBox.critical(self, "Simulation Error", msg)
        self.shoot_btn.setEnabled(True)
        self.shoot_btn.setText("☢   SHOOT X-RAY")
        self.progress.setVisible(False)
        self.statusBar().showMessage("Simulation failed — see error dialog.")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette fallback
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor("#0e0e1c"))
    pal.setColor(QPalette.WindowText,      QColor("#c8cce8"))
    pal.setColor(QPalette.Base,            QColor("#161628"))
    pal.setColor(QPalette.AlternateBase,   QColor("#0e0e1c"))
    pal.setColor(QPalette.ToolTipBase,     QColor("#161628"))
    pal.setColor(QPalette.ToolTipText,     QColor("#c8cce8"))
    pal.setColor(QPalette.Text,            QColor("#c8cce8"))
    pal.setColor(QPalette.Button,          QColor("#161628"))
    pal.setColor(QPalette.ButtonText,      QColor("#c8cce8"))
    pal.setColor(QPalette.BrightText,      QColor("#ffffff"))
    pal.setColor(QPalette.Highlight,       QColor("#2a4080"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
