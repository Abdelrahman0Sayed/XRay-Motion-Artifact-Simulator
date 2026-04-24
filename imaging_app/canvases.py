"""Matplotlib canvases for 3-D phantom and 2-D projection results."""

import numpy as np
from scipy import ndimage

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PyQt5.QtWidgets import QSizePolicy

from .constants import BODY_PARTS, PROJ_AXES


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

    def _build_point_cloud(self):
        ph = self.phantom
        nx, ny, nz = ph.shape
        xi, yi, zi = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        coords = np.stack([xi.ravel(), yi.ravel(), zi.ravel()], axis=1)
        vals = ph.ravel()
        rng = np.random.default_rng(7)

        pts, cols = [], []

        def sample(mask_flat, color, frac):
            idx = np.where(mask_flat)[0]
            if len(idx) == 0:
                return
            n = max(1, int(len(idx) * frac))
            chosen = rng.choice(idx, min(n, len(idx)), replace=False)
            pts.append(coords[chosen])
            cols.extend([color] * len(chosen))

        bone_m = vals > 0.45
        lung_m = (vals > 0.01) & (vals < 0.10)
        organ_m = (vals > 0.21) & (vals < 0.40)

        soft_all = ((vals > 0.13) & (vals < 0.45)).reshape(ph.shape)
        soft_surf = soft_all & ~ndimage.binary_erosion(soft_all)
        soft_flat = soft_surf.ravel()

        sample(bone_m, "#e8e8e8", 0.22)
        sample(soft_flat, "#b07850", 0.14)
        sample(lung_m, "#6aadcc", 0.14)
        sample(organ_m & ~soft_flat, "#bb3333", 0.10)

        self._pc = np.vstack(pts) if pts else np.zeros((1, 3))
        self._pc_colors = cols if cols else ["#666666"]

    def _draw_base(self):
        ax = self._ax
        ax.cla()
        ax.set_facecolor(self.BG)

        c = self._pc
        ax.scatter(
            c[:, 0],
            c[:, 2],
            c[:, 1],
            c=self._pc_colors,
            s=1.2,
            alpha=0.65,
            depthshade=True,
            rasterized=True,
        )

        ax.set_xlabel("X", color="#5566aa", fontsize=7, labelpad=2)
        ax.set_ylabel("Z  (Sup->)", color="#5566aa", fontsize=7, labelpad=2)
        ax.set_zlabel("Y  (Depth)", color="#5566aa", fontsize=7, labelpad=2)
        ax.tick_params(colors="#334466", labelsize=5)
        ax.set_title("3-D Phantom", color="#8899cc", fontsize=9, pad=3)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#1a1a38")
        ax.view_init(elev=15, azim=-65)
        self.draw_idle()

    def update_tube(self, body_part, proj_direction):
        ax = self._ax

        for artist in self._tube_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._tube_artists.clear()

        nx, ny, nz = self.phantom.shape
        cx, cy = nx // 2, ny // 2

        z_norm = BODY_PARTS.get(body_part, (-1.0, 1.0))
        z_lo = int((z_norm[0] + 1) / 2 * nz)
        z_hi = int((z_norm[1] + 1) / 2 * nz)
        z_ctr = (z_lo + z_hi) // 2
        dz = max((z_hi - z_lo) // 3, 4)

        axis_idx = PROJ_AXES.get(proj_direction, 1)
        beam_color = "#ffe044"
        src_color = "#ffcc00"
        det_color = "#44eebb"

        if axis_idx == 1:
            y_src, y_det = -14, ny + 14
            for dxi in (-nx // 5, nx // 5):
                for dzi in (-dz, dz):
                    ln, = ax.plot(
                        [cx + dxi, cx],
                        [z_ctr + dzi, z_ctr],
                        [y_src, cy],
                        color=beam_color,
                        alpha=0.28,
                        lw=0.8,
                        linestyle="--",
                    )
                    self._tube_artists.append(ln)
            sc = ax.scatter([cx], [z_ctr], [y_src], c=src_color, s=90, marker="*", zorder=8, depthshade=False)
            dc = ax.scatter([cx], [z_ctr], [y_det], c=det_color, s=45, marker="s", zorder=8, depthshade=False)
        else:
            x_src, x_det = -14, nx + 14
            for dzi in (-dz, dz):
                ln, = ax.plot(
                    [x_src, cx],
                    [z_ctr + dzi, z_ctr],
                    [cy, cy],
                    color=beam_color,
                    alpha=0.28,
                    lw=0.8,
                    linestyle="--",
                )
                self._tube_artists.append(ln)
            sc = ax.scatter([x_src], [z_ctr], [cy], c=src_color, s=90, marker="*", zorder=8, depthshade=False)
            dc = ax.scatter([x_det], [z_ctr], [cy], c=det_color, s=45, marker="s", zorder=8, depthshade=False)

        tgt = ax.scatter([cx], [z_ctr], [cy], c="#ff3322", s=60, marker="o", alpha=0.80, zorder=9, depthshade=False)
        self._tube_artists += [sc, dc, tgt]

        for zv in (z_lo, z_hi):
            bx, = ax.plot([0, nx, nx, 0, 0], [zv, zv, zv, zv, zv], [0, 0, ny, ny, 0], color="#ff3322", alpha=0.18, lw=0.7)
            self._tube_artists.append(bx)

        self.draw_idle()


class Projection2DCanvas(FigureCanvas):
    """Three-panel display: static | motion blur | mitigated."""

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
        ax.text(0.5, 0.52, "No exposure yet.", ha="center", va="center", color="#3a4a6a", fontsize=13, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.42, "Configure parameters and press  SHOOT X-RAY", ha="center", va="center", color="#2a3a58", fontsize=9, transform=ax.transAxes)
        ax.axis("off")
        self.draw_idle()

    def show_results(self, static, motion, mitigated, params):
        self._last_results = (static, motion, mitigated, params)
        self.fig.clf()
        cmap = "gray"
        titles = [
            "1) Static (no motion)",
            "2) Motion Artifact\n" + motion_label(params),
            f"3) Mitigated\n({params.mitigation})",
        ]
        images = [static, motion, mitigated]
        colors = ["#66aaff", "#ff6644", "#44ee88"]

        for k, (img, ttl, col) in enumerate(zip(images, titles, colors)):
            ax = self.fig.add_subplot(1, 3, k + 1)
            ax.set_facecolor("#000008")
            display_img = (1.0 - img) if self.film_mode else img
            ax.imshow(display_img.T, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="bilinear", origin="lower")
            if self.show_centerlines:
                h, w = display_img.T.shape
                x_mid = (w - 1) / 2.0
                y_mid = (h - 1) / 2.0
                ax.axvline(x_mid, color="#ffde59", linestyle="--", linewidth=0.8, alpha=0.9)
                ax.axhline(y_mid, color="#ffde59", linestyle="--", linewidth=0.8, alpha=0.9)
            ax.set_title(ttl, color=col, fontsize=7.5, pad=5, fontweight="bold")
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(col)
                spine.set_linewidth(1.2)
                spine.set_visible(True)

        self.fig.suptitle(
            f"Exposure {params.exposure_time:.2f} s  -  N0 = {params.n_photons:,}",
            color="#5566aa",
            fontsize=7.5,
            y=0.01,
        )
        self.fig.tight_layout(rect=[0, 0.05, 1, 0.97])
        self.draw_idle()


def motion_label(params):
    mt = params.motion_type
    if mt == "none":
        return "(no motion)"
    if mt == "linear":
        return f"linear  v={params.velocity:.1f} cm/s"
    if mt == "breathing":
        return f"breathing  A={params.amplitude:.1f} cm  f={params.frequency:.2f} Hz"
    if mt == "cardiac":
        return f"cardiac  A={params.amplitude:.1f} cm  f={params.frequency:.2f} Hz"
    return mt
