"""Main application window and UI behavior."""

import numpy as np
from dataclasses import replace
from pubsub import pub

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .canvases import Phantom3DCanvas, ProjectionImageCanvas, motion_label
from .constants import APP_STYLE, BODY_PARTS, PROJ_AXES
from .messaging import topics
from .messaging.message_types import (
    SimulationDoneMessage,
    SimulationErrorMessage,
    SimulationParams,
    SimulationProgressMessage,
)
from .phantom import build_phantom
from .worker import SimulationWorker
from .physics import (
    apply_mitigation,
    compute_nmse,
    compute_ssim,
)

class MainWindow(QMainWindow):
    sig_sim_progress = pyqtSignal(int)
    sig_sim_done = pyqtSignal(object)
    sig_sim_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("  X-Ray Motion Artifact Simulator  -  SBE 4220")
        self.setMinimumSize(1380, 780)
        self.setStyleSheet(APP_STYLE)

        self._sim_worker = None
        self._exp_val = 0.50
        self._vel_val = 1.50
        self._amp_val = 1.50
        self._freq_val = 0.30
        self._last_static = None
        self._last_motion = None
        self._last_params = None
        self.sig_sim_progress.connect(self._do_sim_progress)
        self.sig_sim_done.connect(self._do_sim_done)
        self.sig_sim_error.connect(self._do_sim_error)

        self._register_pubsub()

        self.statusBar().showMessage("Building 3-D phantom... please wait")
        self.phantom = build_phantom((64, 64, 120))

        self._build_ui()
        self._on_motion_type_changed(self.cb_motion.currentText())
        self._refresh_tube()
        self.statusBar().showMessage("Phantom ready. Adjust parameters and press SHOOT X-RAY")

    def _register_pubsub(self):
        pub.subscribe(self._on_sim_requested, topics.SIM_REQUESTED)
        pub.subscribe(self._on_sim_progress, topics.SIM_PROGRESS)
        pub.subscribe(self._on_sim_done, topics.SIM_DONE)
        pub.subscribe(self._on_sim_error, topics.SIM_ERROR)

    def closeEvent(self, event):
        pub.unsubscribe(self._on_sim_requested, topics.SIM_REQUESTED)
        pub.unsubscribe(self._on_sim_progress, topics.SIM_PROGRESS)
        pub.unsubscribe(self._on_sim_done, topics.SIM_DONE)
        pub.unsubscribe(self._on_sim_error, topics.SIM_ERROR)
        super().closeEvent(event)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        controls_box = QFrame()
        controls_box.setStyleSheet("QFrame { border: 1px solid #1a2648; border-radius:8px; }")
        controls_box.setMinimumWidth(300)
        controls_box.setMaximumWidth(305)
        controls_lay = QVBoxLayout(controls_box)
        controls_lay.setContentsMargins(2, 2, 2, 2)

        lbl_controls = QLabel("  CONTROL PANEL")
        lbl_controls.setObjectName("header")
        lbl_controls.setStyleSheet("background:#0f1428; color:#4466aa; padding:6px; font-size:10px; letter-spacing:1px; border-radius:4px;")
        controls_lay.addWidget(lbl_controls)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(0)
        scroll.setMaximumWidth(16777215)
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

        self.shoot_btn = QPushButton("SHOOT X-RAY")
        self.shoot_btn.setObjectName("shoot_btn")
        self.shoot_btn.clicked.connect(self._on_shoot)
        ctrl_lay.addWidget(self.shoot_btn)

        self.btn_mitigate = QPushButton("MITIGATE MOTION")
        self.btn_mitigate.setStyleSheet("""
            QPushButton {
                background: #182441; border: 1px solid #30508a; border-radius: 6px;
                color: #a9c4ff; font-size: 13px; font-weight: bold; padding: 10px; margin-top: 5px;
            }
            QPushButton:hover { border-color: #5f9fff; background: #254f95; }
            QPushButton:disabled { background: #14182a; color: #444466; border-color: #2a2a44; }
        """)
        self.btn_mitigate.clicked.connect(self._on_mitigate_clicked)
        self.btn_mitigate.setEnabled(False)
        ctrl_lay.addWidget(self.btn_mitigate)

        ctrl_lay.addStretch(1)
        scroll.setWidget(ctrl_w)
        controls_lay.addWidget(scroll)

        root.addWidget(controls_box)

        content_box = QFrame()
        content_box.setStyleSheet("QFrame { border: 1px solid #1a2648; border-radius:8px; }")
        content_lay = QVBoxLayout(content_box)
        content_lay.setContentsMargins(3, 3, 3, 3)

        viewer_grid = QGridLayout()
        viewer_grid.setSpacing(8)

        panel_static, lay_static = self._create_viewer_panel("  STATIC IMAGE")
        self.canvas_static = ProjectionImageCanvas()
        lay_static.addWidget(self.canvas_static)

        panel_motion, lay_motion = self._create_viewer_panel("  MOTION ARTIFACT")
        self.canvas_motion = ProjectionImageCanvas()
        lay_motion.addWidget(self.canvas_motion)

        panel_mitig, lay_mitig = self._create_viewer_panel("  MITIGATED RESULT")
        self.canvas_mitig = ProjectionImageCanvas()
        lay_mitig.addWidget(self.canvas_mitig)

        panel_3d, lay_3d = self._create_viewer_panel("  3-D PHANTOM VIEWER")
        self.canvas3d = Phantom3DCanvas(self.phantom)
        lay_3d.addWidget(self.canvas3d)

        leg_row = QHBoxLayout()
        for txt, col in [
            ("o Bone", "#e8e8e8"),
            ("o Tissue", "#b07850"),
            ("o Lung", "#6aadcc"),
            ("o Organ", "#bb3333"),
            ("* Source", "#ffcc00"),
            ("[] Detector", "#44eebb"),
        ]:
            lb = QLabel(txt)
            lb.setStyleSheet(f"color:{col}; font-size:9px;")
            leg_row.addWidget(lb)
        lay_3d.addLayout(leg_row)

        viewer_grid.addWidget(panel_static, 0, 0)
        viewer_grid.addWidget(panel_motion, 0, 1)
        viewer_grid.addWidget(panel_mitig, 1, 0)
        viewer_grid.addWidget(panel_3d, 1, 1)
        viewer_grid.setColumnStretch(0, 1)
        viewer_grid.setColumnStretch(1, 1)
        viewer_grid.setRowStretch(0, 1)
        viewer_grid.setRowStretch(1, 1)

        content_lay.addLayout(viewer_grid)

        metrics = QHBoxLayout()
        self.lbl_nmse_m = QLabel("Motion NMSE: -")
        self.lbl_nmse_r = QLabel("Mitigated NMSE: -")
        self.lbl_ssim_m = QLabel("Motion SSIM: -")      # Added Motion SSIM label
        self.lbl_ssim_r = QLabel("Mitigated SSIM: -")
        
        for lb in (self.lbl_nmse_m, self.lbl_nmse_r, self.lbl_ssim_m, self.lbl_ssim_r):
            lb.setStyleSheet("color:#4488dd; font-size:10px; font-weight:600;")
            metrics.addWidget(lb)
        content_lay.addLayout(metrics)

        root.addWidget(content_box, stretch=1)

    def _create_viewer_panel(self, title):
        panel = QFrame()
        panel.setStyleSheet("QFrame { border: 1px solid #1a2648; border-radius:8px; }")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(3, 3, 3, 3)

        hdr = QLabel(title)
        hdr.setObjectName("header")
        hdr.setStyleSheet("background:#0f1428; color:#4466aa; padding:6px; font-size:10px; letter-spacing:1px; border-radius:4px;")
        lay.addWidget(hdr)

        return panel, lay

    def _grp_target(self):
        g = QGroupBox("TARGET & DIRECTION")
        lay = QGridLayout(g)
        lay.setSpacing(5)

        lay.addWidget(QLabel("Body Part:"), 0, 0)
        self.cb_part = QComboBox()
        self.cb_part.addItems(list(BODY_PARTS.keys()))
        self.cb_part.setCurrentText("Chest")
        self.cb_part.currentTextChanged.connect(self._refresh_tube)
        lay.addWidget(self.cb_part, 0, 1)

        lay.addWidget(QLabel("Projection:"), 1, 0)
        self.cb_proj = QComboBox()
        self.cb_proj.addItems(list(PROJ_AXES.keys()))
        self.cb_proj.currentTextChanged.connect(self._refresh_tube)
        lay.addWidget(self.cb_proj, 1, 1)

        lay.addWidget(QLabel("Motion Axis:"), 2, 0)
        self.cb_maxis = QComboBox()
        self.cb_maxis.addItems(["X  (Left-Right)", "Z  (Head-Foot)"])
        lay.addWidget(self.cb_maxis, 2, 1)

        return g

    def _slider_row(self, label, lo, hi, init, dec, unit, attr):
        w = QWidget()
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

        sl = QSlider(1)
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
        g = QGroupBox("X-RAY PARAMETERS")
        lay = QVBoxLayout(g)
        lay.setSpacing(5)
        lay.addWidget(self._slider_row("Exposure (s):", 0.01, 3.0, 0.5, 2, "s", "_exp_val"))

        row = QHBoxLayout()
        row.addWidget(QLabel("Photon Flux N0:"))
        self.cb_flux = QComboBox()
        for n, lbl in [
            (500, "500  (extreme low)"),
            (2000, "2 k  (very low)"),
            (10000, "10 k  (low)"),
            (50000, "50 k  (medium)"),
            (200000, "200 k  (high)"),
            (1000000, "1 M  (very high)"),
        ]:
            self.cb_flux.addItem(lbl, n)
        self.cb_flux.setCurrentIndex(3)
        row.addWidget(self.cb_flux)
        lay.addLayout(row)
        return g

    def _grp_motion(self):
        g = QGroupBox("MOTION PARAMETERS")
        lay = QVBoxLayout(g)
        lay.setSpacing(5)

        row = QHBoxLayout()
        row.addWidget(QLabel("Motion Type:"))
        self.cb_motion = QComboBox()
        self.cb_motion.addItems(["none", "linear", "breathing"])
        self.cb_motion.setCurrentText("breathing")
        self.cb_motion.currentTextChanged.connect(self._on_motion_type_changed)
        row.addWidget(self.cb_motion)
        lay.addLayout(row)

        self._w_vel = self._slider_row("Velocity:", 0.0, 15.0, 1.5, 2, "cm/s", "_vel_val")
        self._w_amp = self._slider_row("Amplitude:", 0.0, 5.0, 1.5, 2, "cm", "_amp_val")
        self._w_freq = self._slider_row("Frequency:", 0.1, 3.0, 0.3, 2, "Hz", "_freq_val")
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
        g = QGroupBox("MITIGATION STRATEGY")
        lay = QHBoxLayout(g)
        lay.addWidget(QLabel("Method:"))
        self.cb_mitig = QComboBox()
        self.cb_mitig.addItems([
            "None",
            "Unsharp Mask",
            "RL Deconvolution",
        ])
        self.cb_mitig.setCurrentText("RL Deconvolution")
        lay.addWidget(self.cb_mitig)
        return g

    def _grp_view(self):
        g = QGroupBox("RESULT VIEW")
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

    def _on_motion_type_changed(self, mtype):
        lin = mtype == "linear"
        sin = mtype == "breathing"
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
            self.lbl_motion_hint.setText("Velocity and direction are disabled in breathing.")

    def _set_slider_row_enabled(self, row_widget, label_widget, enabled):
        row_widget.setEnabled(enabled)
        row_widget.setToolTip("" if enabled else "Disabled for selected motion type")
        base = label_widget.property("base_text") or label_widget.text()
        label_widget.setText(base if enabled else f"{base} (disabled)")

    def _on_view_option_changed(self):
        enabled_film = self.btn_film_mode.isChecked()
        enabled_lines = self.btn_centerlines.isChecked()
        for canvas in (self.canvas_static, self.canvas_motion, self.canvas_mitig):
            canvas.set_film_mode(enabled_film)
            canvas.set_centerlines_enabled(enabled_lines)

    def _refresh_tube(self):
        if hasattr(self, "canvas3d"):
            self.canvas3d.update_tube(self.cb_part.currentText(), self.cb_proj.currentText())

    def _collect_params(self):
        mtype = self.cb_motion.currentText()
        maxis = 0 if "X" in self.cb_maxis.currentText() else 2
        if mtype != "linear":
            maxis = 2
        return SimulationParams(
            body_part=self.cb_part.currentText(),
            proj_axis=PROJ_AXES[self.cb_proj.currentText()],
            exposure_time=self._exp_val,
            n_photons=self.cb_flux.currentData(),
            motion_type=mtype,
            velocity=self._vel_val if mtype == "linear" else 0.0,
            amplitude=self._amp_val,
            frequency=self._freq_val,
            motion_axis=maxis,
            n_steps=self.sb_steps.value(),
            noise_type="None",
            mitigation=self.cb_mitig.currentText(),
        )

    def _on_shoot(self):
        if self._sim_worker and self._sim_worker.is_alive():
            return

        params = self._collect_params()
        self.shoot_btn.setEnabled(False)
        self.shoot_btn.setText("Simulating...")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.statusBar().showMessage(
            f"Simulating... {params.motion_type} motion | {params.n_steps} integration steps | noise disabled"
        )

        pub.sendMessage(topics.SIM_REQUESTED, message=params)

    def _on_sim_requested(self, message: SimulationParams):
        self._sim_worker = SimulationWorker(self.phantom, message)
        self._sim_worker.start()

    def _on_sim_progress(self, message: SimulationProgressMessage):
        self.sig_sim_progress.emit(message.value)

    def _do_sim_progress(self, value):
        self.progress.setValue(value)

    def _on_sim_done(self, message: SimulationDoneMessage):
        self.sig_sim_done.emit(message)

    def _do_sim_done(self, message: SimulationDoneMessage):

        self._last_static = message.static
        self._last_motion = message.motion
        self._last_params = message.params
        
        self.btn_mitigate.setEnabled(True)
        self.canvas_static.show_image(message.static, "Static (no motion)", "#66aaff")
        self.canvas_motion.show_image(message.motion, f"Motion Artifact - {motion_label(message.params)}", "#ff6644")
        self.canvas_mitig.show_image(message.mitigated, f"Mitigated ({message.params.mitigation})", "#44ee88")

        nmse_m = message.metrics.get("nmse_motion", float("nan"))
        nmse_r = message.metrics.get("nmse_mitigated", float("nan"))
        ssim_m = message.metrics.get("ssim_motion", float("nan"))  # Extract Motion SSIM
        ssim_r = message.metrics.get("ssim_mitig", float("nan"))

        def fmt_val(v):
            return f"{v:.4f}" if np.isfinite(v) else "-"

        self.lbl_nmse_m.setText(f"Motion NMSE: {fmt_val(nmse_m)}")
        self.lbl_nmse_r.setText(f"Mitigated NMSE: {fmt_val(nmse_r)}")
        self.lbl_ssim_m.setText(f"Motion SSIM: {fmt_val(ssim_m)}")  # Display Motion SSIM
        self.lbl_ssim_r.setText(f"Mitigated SSIM: {fmt_val(ssim_r)}")

        self.shoot_btn.setEnabled(True)
        self.shoot_btn.setText("SHOOT X-RAY")
        self.progress.setVisible(False)

        # For NMSE, a lower value indicates improvement
        delta = nmse_m - nmse_r if np.isfinite(nmse_m) and np.isfinite(nmse_r) else 0
        arrow = "down" if delta >= 0 else "up"
        self.statusBar().showMessage(
            f"Done. Motion NMSE = {fmt_val(nmse_m)} -> After mitigation = {fmt_val(nmse_r)} ({arrow} {abs(delta):.4f})"
        )

    def _on_mitigate_clicked(self):
        if self._last_static is None or self._last_motion is None:
            return

        # Apply mitigation locally using a fresh params copy because SimulationParams is frozen
        selected_method = self.cb_mitig.currentText()
        params_for_mitigation = replace(self._last_params, mitigation=selected_method)

        mitigated = apply_mitigation(self._last_motion, params_for_mitigation)

        # Compute new metrics
        nmse_r = compute_nmse(self._last_static, mitigated)
        ssim_r = compute_ssim(self._last_static, mitigated)

        # Update UI components
        def fmt_val(v):
            return f"{v:.4f}" if np.isfinite(v) else "-"

        self.canvas_mitig.show_image(mitigated, f"Mitigated ({selected_method})", "#44ee88")
        self.lbl_nmse_r.setText(f"Mitigated NMSE: {fmt_val(nmse_r)}")
        self.lbl_ssim_r.setText(f"Mitigated SSIM: {fmt_val(ssim_r)}")

        # Calculate original motion NMSE to show delta in status bar
        nmse_m = compute_nmse(self._last_static, self._last_motion)
        delta = nmse_m - nmse_r if np.isfinite(nmse_m) and np.isfinite(nmse_r) else 0
        arrow = "down" if delta >= 0 else "up"
        self.statusBar().showMessage(
            f"Instant Mitigation Applied. Motion NMSE = {fmt_val(nmse_m)} -> Mitigated = {fmt_val(nmse_r)} ({arrow} {abs(delta):.4f})"
        )
    def _on_sim_error(self, message: SimulationErrorMessage):
        self.sig_sim_error.emit(message.traceback_text)

    def _do_sim_error(self, traceback_text):
        QMessageBox.critical(self, "Simulation Error", traceback_text)
        self.shoot_btn.setEnabled(True)
        self.shoot_btn.setText("SHOOT X-RAY")
        self.progress.setVisible(False)
        self.statusBar().showMessage("Simulation failed - see error dialog.")
