"""
=============================================================================
Numerical Simulation of Motion Artifacts in Planar X-Ray Acquisition
Medical Imaging II (SBE 4220) — Spring 2026
Team: Abdelrahman Sayed, Ahmed Raafat, Salah Mohamed, Anas Mohamed, Ahmed Adil
=============================================================================

Requirements:
    pip install PyQt5 numpy scipy scikit-image matplotlib

Run:
    python xray_motion_simulator.py
"""

import sys
import numpy as np
from scipy.ndimage import shift as nd_shift
from scipy.signal import wiener
from scipy.ndimage import uniform_filter, gaussian_filter
import warnings
warnings.filterwarnings("ignore")

# ── Matplotlib with Qt5 backend ───────────────────────────────────────────────
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── scikit-image ──────────────────────────────────────────────────────────────
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import wiener as sk_wiener

# ── PyQt5 ─────────────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QSlider, QComboBox, QPushButton, QGroupBox,
    QTabWidget, QSplitter, QProgressBar, QStatusBar, QFrame,
    QDoubleSpinBox, QSpinBox, QCheckBox, QScrollArea, QSizePolicy,
    QTextEdit, QToolButton
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon


# =============================================================================
#  COLOUR PALETTE  (dark scientific theme)
# =============================================================================
DARK_BG      = "#0d1117"
PANEL_BG     = "#161b22"
BORDER_COL   = "#30363d"
ACCENT_BLUE  = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_ORANGE= "#f0883e"
ACCENT_RED   = "#f85149"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED   = "#8b949e"
HIGHLIGHT    = "#21262d"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', 'SF Pro Text', Arial, sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    border: 1px solid {BORDER_COL};
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 8px;
    font-weight: 600;
    color: {ACCENT_BLUE};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
QLabel {{ color: {TEXT_PRIMARY}; }}
QLabel#muted {{ color: {TEXT_MUTED}; font-size: 11px; }}
QLabel#metric {{ color: {ACCENT_GREEN}; font-weight: bold; font-size: 12px; }}
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER_COL};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT_BLUE};
    border: 2px solid {ACCENT_BLUE};
    width: 14px; height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT_BLUE}; border-radius: 2px; }}
QPushButton {{
    background-color: {ACCENT_BLUE};
    color: {DARK_BG};
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: 700;
    font-size: 13px;
}}
QPushButton:hover {{ background-color: #79b8ff; }}
QPushButton:pressed {{ background-color: #388bfd; }}
QPushButton#secondary {{
    background-color: {HIGHLIGHT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COL};
}}
QPushButton#secondary:hover {{ background-color: {BORDER_COL}; }}
QPushButton#danger {{
    background-color: {ACCENT_RED};
    color: white;
}}
QComboBox {{
    background-color: {HIGHLIGHT};
    border: 1px solid {BORDER_COL};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT_PRIMARY};
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background-color: {PANEL_BG};
    selection-background-color: {ACCENT_BLUE};
    color: {TEXT_PRIMARY};
}}
QDoubleSpinBox, QSpinBox {{
    background-color: {HIGHLIGHT};
    border: 1px solid {BORDER_COL};
    border-radius: 4px;
    padding: 3px 6px;
    color: {TEXT_PRIMARY};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER_COL};
    background-color: {PANEL_BG};
    border-radius: 0 6px 6px 6px;
}}
QTabBar::tab {{
    background-color: {HIGHLIGHT};
    color: {TEXT_MUTED};
    padding: 8px 18px;
    border: 1px solid {BORDER_COL};
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {PANEL_BG};
    color: {ACCENT_BLUE};
    border-bottom: 2px solid {ACCENT_BLUE};
}}
QProgressBar {{
    border: 1px solid {BORDER_COL};
    border-radius: 4px;
    background-color: {HIGHLIGHT};
    text-align: center;
    color: {TEXT_PRIMARY};
    height: 18px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT_BLUE};
    border-radius: 3px;
}}
QStatusBar {{
    background-color: {PANEL_BG};
    border-top: 1px solid {BORDER_COL};
    color: {TEXT_MUTED};
}}
QTextEdit {{
    background-color: {HIGHLIGHT};
    border: 1px solid {BORDER_COL};
    border-radius: 4px;
    color: {TEXT_PRIMARY};
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}}
QScrollBar:vertical {{
    background: {HIGHLIGHT};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_COL};
    border-radius: 4px;
    min-height: 20px;
}}
QCheckBox {{ color: {TEXT_PRIMARY}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {BORDER_COL};
    border-radius: 3px;
    background: {HIGHLIGHT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT_BLUE};
    border-color: {ACCENT_BLUE};
}}
QSplitter::handle {{
    background: {BORDER_COL};
}}
"""


# =============================================================================
#  PHYSICS / SIMULATION ENGINE
# =============================================================================

def create_phantom(size: int = 256, phantom_type: str = "shepp_logan") -> np.ndarray:
    """Create a digital phantom (attenuation map)."""
    if phantom_type == "shepp_logan":
        ph = shepp_logan_phantom()
        ph = resize(ph, (size, size), anti_aliasing=True)
    elif phantom_type == "chest":
        ph = _synthetic_chest(size)
    elif phantom_type == "head":
        ph = _synthetic_head(size)
    else:
        ph = shepp_logan_phantom()
        ph = resize(ph, (size, size), anti_aliasing=True)
    return np.clip(ph, 0, 1).astype(np.float32)


def _synthetic_chest(size: int) -> np.ndarray:
    """Simple synthetic chest phantom."""
    ph = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    # Thorax outline
    Y, X = np.ogrid[:size, :size]
    ellipse = ((X - cx) / (size * 0.42))**2 + ((Y - cy) / (size * 0.38))**2
    ph[ellipse <= 1] = 0.2
    # Lungs (low attenuation)
    for lx in [cx - size // 6, cx + size // 6]:
        lung = ((X - lx) / (size * 0.15))**2 + ((Y - (cy + size//12)) / (size * 0.22))**2
        ph[lung <= 1] = 0.05
    # Heart
    heart = ((X - (cx - size//15)) / (size * 0.12))**2 + ((Y - (cy + size//12)) / (size * 0.10))**2
    ph[heart <= 1] = 0.35
    # Spine
    spine = ((X - cx) / (size * 0.025))**2 + ((Y - cy) / (size * 0.3))**2
    ph[spine <= 1] = 0.7
    # Ribs
    for rib_y in np.linspace(cy - size//6, cy + size//5, 8):
        rib = ((X - cx) / (size * 0.38))**2 + ((Y - rib_y) / (size * 0.015))**2
        mask = rib <= 1
        rib_x_mask = np.abs(X - cx) > size * 0.12
        ph[mask & rib_x_mask] = 0.55
    return gaussian_filter(ph, sigma=1.5)


def _synthetic_head(size: int) -> np.ndarray:
    """Simple synthetic head phantom."""
    ph = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    # Skull
    skull_out = ((X - cx) / (size * 0.42))**2 + ((Y - cy) / (size * 0.48))**2
    skull_in  = ((X - cx) / (size * 0.38))**2 + ((Y - cy) / (size * 0.44))**2
    ph[skull_out <= 1] = 0.6
    ph[skull_in  <= 1] = 0.15   # brain tissue
    # Ventricles
    vent = ((X - cx) / (size * 0.08))**2 + ((Y - cy) / (size * 0.06))**2
    ph[vent <= 1] = 0.03
    # Eyes
    for ex in [cx - size//7, cx + size//7]:
        eye = ((X - ex) / (size * 0.05))**2 + ((Y - (cy - size//6)) / (size * 0.04))**2
        ph[eye <= 1] = 0.0
    return gaussian_filter(ph, sigma=1.0)


def beer_lambert_projection(phantom: np.ndarray, mu_scale: float = 1.0) -> np.ndarray:
    """
    Forward projection via Beer-Lambert law.
    I = I0 * exp(-mu * x)  →  for 2D planar: projection = exp(-mu_scale * phantom)
    """
    return np.exp(-mu_scale * phantom).astype(np.float32)


def simulate_linear_motion(
    phantom: np.ndarray,
    velocity_px_per_s: float,
    exposure_time_s: float,
    mu_scale: float = 1.0,
    n_steps: int = 60,
    progress_cb=None
) -> np.ndarray:
    """
    Simulate linear (translational) motion blur.
    The phantom shifts along the x-axis as a linear function of time.
    Final image = (1/T) ∫₀ᵀ proj(phantom shifted by v*t) dt
    """
    total_shift_px = velocity_px_per_s * exposure_time_s
    accumulated = np.zeros_like(phantom, dtype=np.float64)
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        shift_x = total_shift_px * t
        shifted = nd_shift(phantom, [0, shift_x], mode='constant', cval=0)
        accumulated += beer_lambert_projection(shifted, mu_scale)
        if progress_cb:
            progress_cb(int(100 * i / n_steps))
    if progress_cb:
        progress_cb(100)
    return (accumulated / n_steps).astype(np.float32)


def simulate_periodic_motion(
    phantom: np.ndarray,
    amplitude_px: float,
    frequency_hz: float,
    exposure_time_s: float,
    mu_scale: float = 1.0,
    direction: str = "vertical",
    n_steps: int = 100,
    progress_cb=None
) -> np.ndarray:
    """
    Simulate periodic (sinusoidal) motion — models breathing or cardiac motion.
    shift(t) = A * sin(2π * f * t)
    Final image = (1/T) ∫₀ᵀ proj(phantom shifted by shift(t)) dt
    """
    accumulated = np.zeros_like(phantom, dtype=np.float64)
    for i in range(n_steps):
        t = exposure_time_s * i / max(n_steps - 1, 1)
        shift_val = amplitude_px * np.sin(2 * np.pi * frequency_hz * t)
        if direction == "vertical":
            shift_vec = [shift_val, 0]
        elif direction == "horizontal":
            shift_vec = [0, shift_val]
        else:  # both
            shift_vec = [shift_val, shift_val * 0.5]
        shifted = nd_shift(phantom, shift_vec, mode='constant', cval=0)
        accumulated += beer_lambert_projection(shifted, mu_scale)
        if progress_cb:
            progress_cb(int(100 * i / n_steps))
    if progress_cb:
        progress_cb(100)
    return (accumulated / n_steps).astype(np.float32)


def simulate_combined_motion(
    phantom: np.ndarray,
    lin_velocity_px: float,
    amplitude_px: float,
    frequency_hz: float,
    exposure_time_s: float,
    mu_scale: float = 1.0,
    n_steps: int = 100,
    progress_cb=None
) -> np.ndarray:
    """Combined linear + periodic motion."""
    total_shift_x = lin_velocity_px * exposure_time_s
    accumulated = np.zeros_like(phantom, dtype=np.float64)
    for i in range(n_steps):
        t = exposure_time_s * i / max(n_steps - 1, 1)
        shift_x = total_shift_x * (t / exposure_time_s) if exposure_time_s > 0 else 0
        shift_y = amplitude_px * np.sin(2 * np.pi * frequency_hz * t)
        shifted = nd_shift(phantom, [shift_y, shift_x], mode='constant', cval=0)
        accumulated += beer_lambert_projection(shifted, mu_scale)
        if progress_cb:
            progress_cb(int(100 * i / n_steps))
    if progress_cb:
        progress_cb(100)
    return (accumulated / n_steps).astype(np.float32)


def add_poisson_noise(image: np.ndarray, dose_level: float = 1e4) -> np.ndarray:
    """
    Add Poisson photon-counting noise.
    dose_level controls mean photon count (↑ dose = ↓ noise).
    """
    scaled = np.clip(image, 0, None) * dose_level
    noisy = np.random.poisson(scaled).astype(np.float32)
    return noisy / dose_level


def add_gaussian_noise(image: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 1)


def apply_wiener_filter(image: np.ndarray, noise_power: float = 0.01,
                        kernel_size: int = 5) -> np.ndarray:
    """Wiener deconvolution / noise reduction filter."""
    restored = wiener(image, mysize=kernel_size, noise=noise_power)
    return np.clip(restored, 0, 1).astype(np.float32)


def apply_gaussian_sharpening(image: np.ndarray, sigma: float = 1.0,
                               strength: float = 1.5) -> np.ndarray:
    """Unsharp masking for edge enhancement."""
    blurred = gaussian_filter(image, sigma=sigma)
    sharpened = image + strength * (image - blurred)
    return np.clip(sharpened, 0, 1).astype(np.float32)


def apply_total_variation(image: np.ndarray, weight: float = 0.1,
                           iterations: int = 50) -> np.ndarray:
    """Simple gradient-descent TV denoising."""
    u = image.copy().astype(np.float64)
    for _ in range(iterations):
        gy = np.diff(u, axis=0, append=u[-1:])
        gx = np.diff(u, axis=1, append=u[:, -1:])
        norm = np.sqrt(gx**2 + gy**2 + 1e-8)
        div = (np.diff(gx / norm, axis=1, prepend=0) +
               np.diff(gy / norm, axis=0, prepend=0))
        u = u + weight * div
    return np.clip(u, 0, 1).astype(np.float32)


def compute_metrics(reference: np.ndarray, target: np.ndarray) -> dict:
    """Compute image quality metrics."""
    ref = np.clip(reference, 0, 1)
    tgt = np.clip(target, 0, 1)
    try:
        psnr_val = psnr(ref, tgt, data_range=1.0)
    except Exception:
        psnr_val = float('nan')
    try:
        ssim_val = ssim(ref, tgt, data_range=1.0)
    except Exception:
        ssim_val = float('nan')
    mse_val = float(np.mean((ref - tgt) ** 2))
    snr_val = float(10 * np.log10(np.mean(ref**2) / (mse_val + 1e-12)))
    return {"PSNR (dB)": round(psnr_val, 2),
            "SSIM": round(ssim_val, 4),
            "MSE": round(mse_val, 6),
            "SNR (dB)": round(snr_val, 2)}


# =============================================================================
#  WORKER THREAD  (keeps GUI responsive)
# =============================================================================

class SimulationWorker(QThread):
    progress   = pyqtSignal(int)
    finished   = pyqtSignal(dict)
    error      = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            p = self.params
            self.progress.emit(5)

            # 1. Create phantom
            phantom = create_phantom(p["phantom_size"], p["phantom_type"])
            self.progress.emit(10)

            # 2. Static (reference) projection
            static_proj = beer_lambert_projection(phantom, p["mu_scale"])
            self.progress.emit(20)

            def prog(v):
                self.progress.emit(20 + int(v * 0.5))

            # 3. Motion simulation
            motion_type = p["motion_type"]
            if motion_type == "Linear":
                motion_proj = simulate_linear_motion(
                    phantom, p["lin_velocity"], p["exposure_time"],
                    p["mu_scale"], n_steps=p["n_steps"], progress_cb=prog)
            elif motion_type == "Periodic (Breathing)":
                motion_proj = simulate_periodic_motion(
                    phantom, p["amplitude"], p["frequency"], p["exposure_time"],
                    p["mu_scale"], p["direction"], n_steps=p["n_steps"], progress_cb=prog)
            else:  # Combined
                motion_proj = simulate_combined_motion(
                    phantom, p["lin_velocity"], p["amplitude"], p["frequency"],
                    p["exposure_time"], p["mu_scale"], n_steps=p["n_steps"], progress_cb=prog)

            self.progress.emit(72)

            # 4. Add noise
            if p["noise_type"] == "Poisson":
                noisy_proj = add_poisson_noise(motion_proj, p["dose_level"])
                static_noisy = add_poisson_noise(static_proj, p["dose_level"])
            elif p["noise_type"] == "Gaussian":
                sig = 0.5 / np.sqrt(max(p["dose_level"], 1))
                noisy_proj = add_gaussian_noise(motion_proj, sig)
                static_noisy = add_gaussian_noise(static_proj, sig)
            else:
                noisy_proj = motion_proj.copy()
                static_noisy = static_proj.copy()

            self.progress.emit(80)

            # 5. Mitigation
            restored = None
            if p["mitigation"] == "Wiener Filter":
                restored = apply_wiener_filter(noisy_proj, p["wiener_noise"],
                                               p["wiener_ksize"])
            elif p["mitigation"] == "Unsharp Masking":
                restored = apply_gaussian_sharpening(noisy_proj, p["sharp_sigma"],
                                                     p["sharp_strength"])
            elif p["mitigation"] == "Total Variation":
                restored = apply_total_variation(noisy_proj, p["tv_weight"],
                                                 p["tv_iters"])
            elif p["mitigation"] == "Combined (Wiener + Unsharp)":
                tmp = apply_wiener_filter(noisy_proj, p["wiener_noise"], p["wiener_ksize"])
                restored = apply_gaussian_sharpening(tmp, p["sharp_sigma"],
                                                     p["sharp_strength"])

            self.progress.emit(90)

            # 6. Metrics
            metrics_motion   = compute_metrics(static_noisy, noisy_proj)
            metrics_restored = compute_metrics(static_noisy, restored) if restored is not None else None

            self.progress.emit(100)

            self.finished.emit({
                "phantom":        phantom,
                "static_proj":    static_proj,
                "static_noisy":   static_noisy,
                "motion_proj":    motion_proj,
                "noisy_proj":     noisy_proj,
                "restored":       restored,
                "metrics_motion": metrics_motion,
                "metrics_restored": metrics_restored,
                "params":         p,
            })

        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


# =============================================================================
#  MATPLOTLIB CANVAS HELPER
# =============================================================================

class ImgCanvas(FigureCanvas):
    """Single matplotlib figure embedded in PyQt."""

    def __init__(self, parent=None, tight=True):
        self.fig = Figure(facecolor=DARK_BG)
        super().__init__(self.fig)
        if parent:
            self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._tight = tight

    def show_images(self, images: list, titles: list, cmap="gray",
                    suptitle: str = ""):
        """Render a row of images."""
        self.fig.clear()
        n = len(images)
        axes = self.fig.subplots(1, n) if n > 1 else [self.fig.add_subplot(111)]
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation='lanczos')
            ax.set_title(title, color=TEXT_PRIMARY, fontsize=9, pad=4)
            ax.axis("off")
        if suptitle:
            self.fig.suptitle(suptitle, color=ACCENT_BLUE, fontsize=11, y=1.01)
        self.fig.patch.set_facecolor(DARK_BG)
        if self._tight:
            self.fig.tight_layout()
        self.draw()

    def show_diff(self, ref, target, title="Difference Map"):
        """Show absolute difference image."""
        self.fig.clear()
        axes = self.fig.subplots(1, 3)
        axes[0].imshow(ref,    cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("Reference", color=TEXT_PRIMARY, fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(target, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Motion-Blurred", color=TEXT_PRIMARY, fontsize=9)
        axes[1].axis("off")
        diff = np.abs(ref.astype(float) - target.astype(float))
        im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=diff.max() + 1e-9)
        axes[2].set_title("|Difference|", color=TEXT_PRIMARY, fontsize=9)
        axes[2].axis("off")
        self.fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        self.fig.patch.set_facecolor(DARK_BG)
        self.fig.suptitle(title, color=ACCENT_BLUE, fontsize=11)
        self.fig.tight_layout()
        self.draw()

    def show_metrics_chart(self, metrics_motion: dict, metrics_restored: dict = None):
        """Bar chart comparing metrics before and after restoration."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(HIGHLIGHT)
        keys = list(metrics_motion.keys())
        x = np.arange(len(keys))
        width = 0.35
        bars1 = ax.bar(x - width/2, [metrics_motion[k] for k in keys],
                       width, label="Motion-Blurred + Noise",
                       color=ACCENT_ORANGE, alpha=0.85)
        if metrics_restored:
            bars2 = ax.bar(x + width/2, [metrics_restored[k] for k in keys],
                           width, label="After Mitigation",
                           color=ACCENT_GREEN, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, color=TEXT_PRIMARY)
        ax.tick_params(colors=TEXT_PRIMARY)
        ax.set_title("Image Quality Metrics Comparison", color=ACCENT_BLUE, fontsize=11)
        ax.legend(facecolor=PANEL_BG, edgecolor=BORDER_COL, labelcolor=TEXT_PRIMARY)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COL)
        self.fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(HIGHLIGHT)
        self.fig.tight_layout()
        self.draw()

    def show_motion_profile(self, params: dict):
        """Plot the motion trajectory over the exposure window."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(HIGHLIGHT)
        T = params["exposure_time"]
        t = np.linspace(0, T, 500)
        mt = params["motion_type"]

        if mt == "Linear":
            y = params["lin_velocity"] * t
            ax.plot(t, y, color=ACCENT_BLUE, linewidth=2, label="X displacement (px)")
            ax.set_ylabel("Displacement (pixels)", color=TEXT_PRIMARY)
        elif mt == "Periodic (Breathing)":
            y = params["amplitude"] * np.sin(2 * np.pi * params["frequency"] * t)
            ax.plot(t, y, color=ACCENT_GREEN, linewidth=2, label="Y displacement (px)")
            ax.set_ylabel("Displacement (pixels)", color=TEXT_PRIMARY)
        else:
            yx = params["lin_velocity"] * t
            yy = params["amplitude"] * np.sin(2 * np.pi * params["frequency"] * t)
            ax.plot(t, yx, color=ACCENT_BLUE, linewidth=2, label="X (linear)")
            ax.plot(t, yy, color=ACCENT_GREEN, linewidth=2, label="Y (periodic)")
            ax.set_ylabel("Displacement (pixels)", color=TEXT_PRIMARY)

        ax.set_xlabel("Time (s)", color=TEXT_PRIMARY)
        ax.set_title("Motion Profile During Exposure", color=ACCENT_BLUE)
        ax.tick_params(colors=TEXT_PRIMARY)
        ax.legend(facecolor=PANEL_BG, edgecolor=BORDER_COL, labelcolor=TEXT_PRIMARY)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COL)
        ax.axvspan(0, T, alpha=0.08, color=ACCENT_BLUE)
        ax.grid(True, alpha=0.2, color=BORDER_COL)
        self.fig.patch.set_facecolor(DARK_BG)
        self.fig.tight_layout()
        self.draw()


# =============================================================================
#  CONTROL PANEL WIDGETS
# =============================================================================

def _labeled_slider(label: str, lo: int, hi: int, val: int,
                    scale: float = 1.0, unit: str = "") -> tuple:
    """Returns (QGroupBox_row_widget, QSlider, QLabel_value)."""
    row = QWidget()
    hl  = QHBoxLayout(row)
    hl.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label)
    lbl.setFixedWidth(160)
    sl  = QSlider(Qt.Horizontal)
    sl.setRange(lo, hi)
    sl.setValue(val)
    val_lbl = QLabel(f"{val * scale:.2g}{unit}")
    val_lbl.setFixedWidth(60)
    val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    val_lbl.setObjectName("metric")

    def _update(v):
        val_lbl.setText(f"{v * scale:.2g}{unit}")
    sl.valueChanged.connect(_update)

    hl.addWidget(lbl)
    hl.addWidget(sl)
    hl.addWidget(val_lbl)
    return row, sl, val_lbl


# =============================================================================
#  MAIN WINDOW
# =============================================================================

class XRaySimulator(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("X-Ray Motion Artifact Simulator  |  Medical Imaging II (SBE 4220)")
        self.setMinimumSize(1300, 820)
        self.results = None
        self._worker = None
        self._build_ui()
        self.setStyleSheet(STYLESHEET)
        self._run_static_preview()

    # ── UI BUILD ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Left control panel ────────────────────────────────────────────────
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setFixedWidth(320)
        ctrl_scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {DARK_BG}; }}")

        ctrl_inner = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_inner)
        ctrl_layout.setSpacing(10)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        ctrl_scroll.setWidget(ctrl_inner)

        # Title
        title = QLabel("⚕  X-Ray Motion Simulator")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT_BLUE}; padding: 6px 0;")
        ctrl_layout.addWidget(title)

        sub = QLabel("Medical Imaging II — SBE 4220  •  Spring 2026")
        sub.setObjectName("muted")
        sub.setWordWrap(True)
        ctrl_layout.addWidget(sub)

        # ── PHANTOM ───────────────────────────────────────────────────────────
        gb_phantom = QGroupBox("1 · Digital Phantom")
        vp = QVBoxLayout(gb_phantom)

        vp.addWidget(QLabel("Phantom type:"))
        self.cb_phantom = QComboBox()
        self.cb_phantom.addItems(["shepp_logan", "chest", "head"])
        vp.addWidget(self.cb_phantom)

        vp.addWidget(QLabel("Image size:"))
        self.cb_size = QComboBox()
        self.cb_size.addItems(["128", "256", "512"])
        self.cb_size.setCurrentIndex(1)
        vp.addWidget(self.cb_size)

        r, self.sl_mu, _ = _labeled_slider("μ scale:", 5, 30, 10, 0.1, "")
        vp.addWidget(r)
        ctrl_layout.addWidget(gb_phantom)

        # ── MOTION ────────────────────────────────────────────────────────────
        gb_motion = QGroupBox("2 · Motion Model")
        vm = QVBoxLayout(gb_motion)

        vm.addWidget(QLabel("Motion type:"))
        self.cb_motion = QComboBox()
        self.cb_motion.addItems(["Linear", "Periodic (Breathing)", "Combined"])
        self.cb_motion.currentTextChanged.connect(self._on_motion_type_changed)
        vm.addWidget(self.cb_motion)

        r, self.sl_exposure, _ = _labeled_slider("Exposure time:", 1, 500, 50, 0.01, " s")
        vm.addWidget(r)

        self.w_linear = QWidget()
        vl = QVBoxLayout(self.w_linear)
        vl.setContentsMargins(0, 0, 0, 0)
        r, self.sl_velocity, _ = _labeled_slider("Velocity:", 0, 200, 30, 0.5, " px/s")
        vl.addWidget(r)
        vm.addWidget(self.w_linear)

        self.w_periodic = QWidget()
        vper = QVBoxLayout(self.w_periodic)
        vper.setContentsMargins(0, 0, 0, 0)
        r, self.sl_amplitude, _ = _labeled_slider("Amplitude:", 0, 100, 20, 0.5, " px")
        vper.addWidget(r)
        r, self.sl_frequency, _ = _labeled_slider("Frequency:", 1, 50, 5, 0.2, " Hz")
        vper.addWidget(r)
        vm.addWidget(self.w_periodic)

        vm.addWidget(QLabel("Periodic direction:"))
        self.cb_direction = QComboBox()
        self.cb_direction.addItems(["vertical", "horizontal", "both"])
        vm.addWidget(self.cb_direction)

        r, self.sl_steps, _ = _labeled_slider("Integration steps:", 20, 200, 80, 1, "")
        vm.addWidget(r)

        ctrl_layout.addWidget(gb_motion)
        self._on_motion_type_changed("Linear")

        # ── NOISE ─────────────────────────────────────────────────────────────
        gb_noise = QGroupBox("3 · Noise Model")
        vn = QVBoxLayout(gb_noise)

        vn.addWidget(QLabel("Noise type:"))
        self.cb_noise = QComboBox()
        self.cb_noise.addItems(["Poisson", "Gaussian", "None"])
        vn.addWidget(self.cb_noise)

        r, self.sl_dose, _ = _labeled_slider("Dose level:", 100, 100000, 10000, 1, "")
        vn.addWidget(r)

        ctrl_layout.addWidget(gb_noise)

        # ── MITIGATION ────────────────────────────────────────────────────────
        gb_mit = QGroupBox("4 · Mitigation Strategy")
        vmi = QVBoxLayout(gb_mit)

        vmi.addWidget(QLabel("Method:"))
        self.cb_mitigation = QComboBox()
        self.cb_mitigation.addItems([
            "None", "Wiener Filter", "Unsharp Masking",
            "Total Variation", "Combined (Wiener + Unsharp)"
        ])
        vmi.addWidget(self.cb_mitigation)

        # Wiener params
        self.w_wiener = QWidget()
        vw = QVBoxLayout(self.w_wiener)
        vw.setContentsMargins(0, 0, 0, 0)
        r, self.sl_wiener_noise, _ = _labeled_slider("Noise power:", 1, 100, 10, 0.001, "")
        vw.addWidget(r)
        r, self.sl_wiener_k, _ = _labeled_slider("Kernel size:", 3, 21, 5, 2, "px")
        vw.addWidget(r)
        vmi.addWidget(self.w_wiener)

        # Sharpening params
        self.w_sharp = QWidget()
        vs = QVBoxLayout(self.w_sharp)
        vs.setContentsMargins(0, 0, 0, 0)
        r, self.sl_sharp_sigma, _ = _labeled_slider("Sigma:", 1, 30, 10, 0.1, "")
        vs.addWidget(r)
        r, self.sl_sharp_str, _ = _labeled_slider("Strength:", 5, 50, 15, 0.1, "")
        vs.addWidget(r)
        vmi.addWidget(self.w_sharp)

        # TV params
        self.w_tv = QWidget()
        vt = QVBoxLayout(self.w_tv)
        vt.setContentsMargins(0, 0, 0, 0)
        r, self.sl_tv_w, _ = _labeled_slider("TV weight:", 1, 200, 10, 0.001, "")
        vt.addWidget(r)
        r, self.sl_tv_i, _ = _labeled_slider("Iterations:", 10, 200, 50, 1, "")
        vt.addWidget(r)
        vmi.addWidget(self.w_tv)

        ctrl_layout.addWidget(gb_mit)

        # ── BUTTONS ───────────────────────────────────────────────────────────
        self.btn_run = QPushButton("▶  Run Simulation")
        self.btn_run.setFixedHeight(42)
        self.btn_run.clicked.connect(self._run_simulation)
        ctrl_layout.addWidget(self.btn_run)

        self.btn_motion_profile = QPushButton("Plot Motion Profile")
        self.btn_motion_profile.setObjectName("secondary")
        self.btn_motion_profile.clicked.connect(self._show_motion_profile)
        ctrl_layout.addWidget(self.btn_motion_profile)

        self.btn_export = QPushButton("Export Report (PNG)")
        self.btn_export.setObjectName("secondary")
        self.btn_export.clicked.connect(self._export_report)
        ctrl_layout.addWidget(self.btn_export)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        ctrl_layout.addWidget(self.progress_bar)

        ctrl_layout.addStretch()

        # ── Right display area ────────────────────────────────────────────────
        self.tabs = QTabWidget()

        # Tab 1: Overview
        tab_overview = QWidget()
        vov = QVBoxLayout(tab_overview)
        self.canvas_overview = ImgCanvas()
        vov.addWidget(self.canvas_overview)
        self.tabs.addTab(tab_overview, "📷  Overview")

        # Tab 2: Difference map
        tab_diff = QWidget()
        vdiff = QVBoxLayout(tab_diff)
        self.canvas_diff = ImgCanvas()
        vdiff.addWidget(self.canvas_diff)
        self.tabs.addTab(tab_diff, "🔍  Difference Map")

        # Tab 3: Restoration
        tab_rest = QWidget()
        vrest = QVBoxLayout(tab_rest)
        self.canvas_rest = ImgCanvas()
        vrest.addWidget(self.canvas_rest)
        self.tabs.addTab(tab_rest, "✨  Restoration")

        # Tab 4: Metrics chart
        tab_metrics = QWidget()
        vm2 = QVBoxLayout(tab_metrics)
        self.canvas_metrics = ImgCanvas()
        vm2.addWidget(self.canvas_metrics)

        # Metrics text
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(120)
        vm2.addWidget(self.metrics_text)
        self.tabs.addTab(tab_metrics, "📊  Metrics")

        # Tab 5: Motion profile
        tab_profile = QWidget()
        vprof = QVBoxLayout(tab_profile)
        self.canvas_profile = ImgCanvas()
        vprof.addWidget(self.canvas_profile)
        self.tabs.addTab(tab_profile, "📈  Motion Profile")

        # ── Status bar ────────────────────────────────────────────────────────
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage(
            "Ready. Configure parameters and click ▶ Run Simulation.")

        # Assemble root
        root.addWidget(ctrl_scroll)
        root.addWidget(self.tabs, stretch=1)

    # ── SLOT HELPERS ──────────────────────────────────────────────────────────

    def _on_motion_type_changed(self, text: str):
        self.w_linear.setVisible("Linear" in text)
        self.w_periodic.setVisible("Periodic" in text or "Combined" in text)
        self.cb_direction.setVisible("Periodic" in text or "Combined" in text)

    def _get_params(self) -> dict:
        return {
            "phantom_type":   self.cb_phantom.currentText(),
            "phantom_size":   int(self.cb_size.currentText()),
            "mu_scale":       self.sl_mu.value() * 0.1,
            "motion_type":    self.cb_motion.currentText(),
            "exposure_time":  self.sl_exposure.value() * 0.01,
            "lin_velocity":   self.sl_velocity.value() * 0.5,
            "amplitude":      self.sl_amplitude.value() * 0.5,
            "frequency":      self.sl_frequency.value() * 0.2,
            "direction":      self.cb_direction.currentText(),
            "n_steps":        self.sl_steps.value(),
            "noise_type":     self.cb_noise.currentText(),
            "dose_level":     self.sl_dose.value(),
            "mitigation":     self.cb_mitigation.currentText(),
            "wiener_noise":   self.sl_wiener_noise.value() * 0.001,
            "wiener_ksize":   max(3, self.sl_wiener_k.value() * 2 - 1),
            "sharp_sigma":    self.sl_sharp_sigma.value() * 0.1,
            "sharp_strength": self.sl_sharp_str.value() * 0.1,
            "tv_weight":      self.sl_tv_w.value() * 0.001,
            "tv_iters":       self.sl_tv_i.value(),
        }

    # ── SIMULATION ────────────────────────────────────────────────────────────

    def _run_static_preview(self):
        """Show the phantom immediately on launch."""
        phantom = create_phantom(256, "shepp_logan")
        static  = beer_lambert_projection(phantom, 1.0)
        self.canvas_overview.show_images(
            [phantom, static],
            ["Digital Phantom (μ map)", "Static X-Ray Projection"],
            suptitle="Static Preview — Adjust parameters and click ▶ Run Simulation"
        )

    def _run_simulation(self):
        if self._worker and self._worker.isRunning():
            return
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳  Simulating…")
        self.progress_bar.setValue(0)
        self.statusbar.showMessage("Running simulation…")

        params = self._get_params()
        self._worker = SimulationWorker(params)
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.finished.connect(self._on_simulation_done)
        self._worker.error.connect(self._on_simulation_error)
        self._worker.start()

    def _on_simulation_done(self, results: dict):
        self.results = results
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Run Simulation")
        self.statusbar.showMessage("Simulation complete. Rendering results…")

        phantom      = results["phantom"]
        static_noisy = results["static_noisy"]
        motion_proj  = results["motion_proj"]
        noisy_proj   = results["noisy_proj"]
        restored     = results["restored"]
        p            = results["params"]

        # -- Tab 1: Overview ---------------------------------------------------
        imgs   = [phantom, static_noisy, noisy_proj]
        titles = ["Phantom (μ map)", "Reference Projection", f"Motion-Blurred\n({p['motion_type']}, exp={p['exposure_time']:.2f}s)"]
        if restored is not None:
            imgs.append(restored)
            titles.append(f"Restored\n({p['mitigation']})")
        self.canvas_overview.show_images(imgs, titles,
                                         suptitle="X-Ray Acquisition Simulation Pipeline")

        # -- Tab 2: Difference -------------------------------------------------
        self.canvas_diff.show_diff(
            np.clip(static_noisy, 0, 1),
            np.clip(noisy_proj, 0, 1),
            title="Absolute Difference: Reference vs Motion-Blurred"
        )

        # -- Tab 3: Restoration ------------------------------------------------
        if restored is not None:
            rest_imgs = [static_noisy, noisy_proj, restored]
            rest_titles = ["Reference", "Motion + Noise", f"Restored ({p['mitigation']})"]
            self.canvas_rest.show_images(rest_imgs, rest_titles,
                                          suptitle="Mitigation / Restoration Result")
            # Also show diff of restored vs reference
        else:
            self.canvas_rest.show_images(
                [static_noisy, noisy_proj],
                ["Reference", "Motion-Blurred"],
                suptitle="No Mitigation Selected"
            )

        # -- Tab 4: Metrics ----------------------------------------------------
        self.canvas_metrics.show_metrics_chart(
            results["metrics_motion"], results["metrics_restored"])

        mm  = results["metrics_motion"]
        mr  = results["metrics_restored"]
        txt = "=== Motion-Blurred + Noise vs Reference ===\n"
        for k, v in mm.items():
            txt += f"  {k}: {v}\n"
        if mr:
            txt += f"\n=== After Mitigation ({p['mitigation']}) ===\n"
            for k, v in mr.items():
                txt += f"  {k}: {v}\n"
            # Improvement summary
            psnr_diff = mr.get("PSNR (dB)", 0) - mm.get("PSNR (dB)", 0)
            ssim_diff = mr.get("SSIM", 0) - mm.get("SSIM", 0)
            txt += f"\n  PSNR improvement: {psnr_diff:+.2f} dB\n"
            txt += f"  SSIM improvement: {ssim_diff:+.4f}\n"
        self.metrics_text.setPlainText(txt)

        # -- Tab 5: Motion profile ---------------------------------------------
        self.canvas_profile.show_motion_profile(p)

        self.statusbar.showMessage(
            f"Done. PSNR (blurred): {mm['PSNR (dB)']} dB  |  SSIM: {mm['SSIM']}"
            + (f"  →  PSNR (restored): {mr['PSNR (dB)']} dB  |  SSIM: {mr['SSIM']}"
               if mr else "")
        )

    def _on_simulation_error(self, tb: str):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Run Simulation")
        self.metrics_text.setPlainText("ERROR:\n" + tb)
        self.statusbar.showMessage("Simulation error — see Metrics tab for details.")

    # ── EXTRA ACTIONS ─────────────────────────────────────────────────────────

    def _show_motion_profile(self):
        params = self._get_params()
        self.canvas_profile.show_motion_profile(params)
        self.tabs.setCurrentIndex(4)

    def _export_report(self):
        if not self.results:
            self.statusbar.showMessage("No simulation results to export. Run simulation first.")
            return
        r = self.results
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor=DARK_BG)
        fig.suptitle("X-Ray Motion Artifact Simulation Report",
                     color=ACCENT_BLUE, fontsize=16, fontweight='bold')
        imgs_titles = [
            (r["phantom"],     "Phantom (μ map)"),
            (r["static_proj"], "Static Projection\n(Beer-Lambert)"),
            (r["static_noisy"],"Reference + Noise"),
            (r["motion_proj"], f"Motion-Blurred\n({r['params']['motion_type']})"),
        ]
        for ax, (img, title) in zip(axes[0], imgs_titles):
            ax.imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
            ax.set_title(title, color=TEXT_PRIMARY, fontsize=9)
            ax.axis("off")
            ax.set_facecolor(DARK_BG)

        # Diff
        diff = np.abs(r["static_noisy"].astype(float) - r["noisy_proj"].astype(float))
        axes[1][0].imshow(np.clip(r["noisy_proj"], 0, 1), cmap='gray')
        axes[1][0].set_title("Blurred + Noise", color=TEXT_PRIMARY, fontsize=9)
        axes[1][0].axis("off")

        axes[1][1].imshow(diff, cmap='hot')
        axes[1][1].set_title("|Difference|", color=TEXT_PRIMARY, fontsize=9)
        axes[1][1].axis("off")

        if r["restored"] is not None:
            axes[1][2].imshow(np.clip(r["restored"], 0, 1), cmap='gray')
            axes[1][2].set_title(f"Restored\n({r['params']['mitigation']})",
                                 color=TEXT_PRIMARY, fontsize=9)
            axes[1][2].axis("off")
        else:
            axes[1][2].axis("off")

        # Metrics bar
        ax_m = axes[1][3]
        mm = r["metrics_motion"]
        mr = r["metrics_restored"]
        keys = list(mm.keys())
        x = np.arange(len(keys))
        w = 0.35
        ax_m.bar(x - w/2, [mm[k] for k in keys], w,
                 color=ACCENT_ORANGE, label="Blurred")
        if mr:
            ax_m.bar(x + w/2, [mr[k] for k in keys], w,
                     color=ACCENT_GREEN, label="Restored")
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(keys, color=TEXT_PRIMARY, fontsize=8)
        ax_m.legend(facecolor=PANEL_BG, labelcolor=TEXT_PRIMARY, fontsize=8)
        ax_m.set_facecolor(HIGHLIGHT)
        ax_m.set_title("Quality Metrics", color=ACCENT_BLUE, fontsize=9)
        ax_m.tick_params(colors=TEXT_PRIMARY)

        plt.tight_layout()
        out_path = "xray_simulation_report.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=DARK_BG)
        plt.close(fig)
        self.statusbar.showMessage(f"Report exported → {out_path}")


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("X-Ray Motion Artifact Simulator")

    # High-DPI support
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    window = XRaySimulator()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()