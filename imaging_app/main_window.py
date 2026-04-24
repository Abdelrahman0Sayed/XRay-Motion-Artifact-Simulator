"""Main application window and UI behavior."""

import numpy as np
from pubsub import pub

from PyQt5.QtCore import QTimer
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

from .canvases import Phantom3DCanvas, Projection2DCanvas
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


class MainWindow(QMainWindow):
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

        left_box = QFrame()
        left_box.setStyleSheet("QFrame { border: 1px solid #1a2648; border-radius:8px; }")
        left_lay = QVBoxLayout(left_box)
        left_lay.setContentsMargins(3, 3, 3, 3)

        lbl_3d = QLabel("  3-D PHANTOM VIEWER")
        lbl_3d.setObjectName("header")
        lbl_3d.setStyleSheet("background:#0f1428; color:#4466aa; padding:6px; font-size:10px; letter-spacing:1px; border-radius:4px;")
        left_lay.addWidget(lbl_3d)

        self.canvas3d = Phantom3DCanvas(self.phantom)
        left_lay.addWidget(self.canvas3d)

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
        left_lay.addLayout(leg_row)

        root.addWidget(left_box, stretch=38)

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

        self.shoot_btn = QPushButton("SHOOT X-RAY")
        self.shoot_btn.setObjectName("shoot_btn")
        self.shoot_btn.clicked.connect(self._on_shoot)
        ctrl_lay.addWidget(self.shoot_btn)

        ctrl_lay.addStretch(1)
        scroll.setWidget(ctrl_w)
        root.addWidget(scroll, stretch=22)

        right_box = QFrame()
        right_box.setStyleSheet("QFrame { border: 1px solid #1a2648; border-radius:8px; }")
        right_lay = QVBoxLayout(right_box)
        right_lay.setContentsMargins(3, 3, 3, 3)

        lbl_2d = QLabel("  2-D PROJECTION RESULTS")
        lbl_2d.setStyleSheet("background:#0f1428; color:#4466aa; padding:6px; font-size:10px; letter-spacing:1px; border-radius:4px;")
        right_lay.addWidget(lbl_2d)

        self.canvas2d = Projection2DCanvas()
        right_lay.addWidget(self.canvas2d)

        metrics = QHBoxLayout()
        self.lbl_snr_m = QLabel("Motion SNR: -")
        self.lbl_snr_r = QLabel("Mitigated SNR: -")
        self.lbl_psnr_m = QLabel("PSNR: -")
        for lb in (self.lbl_snr_m, self.lbl_snr_r, self.lbl_psnr_m):
            lb.setStyleSheet("color:#4488dd; font-size:10px; font-weight:600;")
            metrics.addWidget(lb)
        right_lay.addLayout(metrics)

        root.addWidget(right_box, stretch=48)

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
        self.cb_motion.addItems(["none", "linear", "breathing", "cardiac"])
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
        QTimer.singleShot(0, lambda v=message.value: self.progress.setValue(v))

    def _on_sim_done(self, message: SimulationDoneMessage):
        def update_ui():
            self.canvas2d.show_results(message.static, message.motion, message.mitigated, message.params)

            snr_m = message.metrics.get("snr_motion", float("nan"))
            snr_r = message.metrics.get("snr_mitigated", float("nan"))
            psnr = message.metrics.get("psnr_mitig", float("nan"))

            def fmt_db(v):
                return f"{v:.1f} dB" if np.isfinite(v) else "-"

            self.lbl_snr_m.setText(f"Motion SNR: {fmt_db(snr_m)}")
            self.lbl_snr_r.setText(f"Mitigated SNR: {fmt_db(snr_r)}")
            self.lbl_psnr_m.setText(f"PSNR: {fmt_db(psnr)}")

            self.shoot_btn.setEnabled(True)
            self.shoot_btn.setText("SHOOT X-RAY")
            self.progress.setVisible(False)

            delta = snr_r - snr_m if np.isfinite(snr_m) and np.isfinite(snr_r) else 0
            arrow = "up" if delta >= 0 else "down"
            self.statusBar().showMessage(
                f"Done. Motion SNR = {fmt_db(snr_m)} -> After mitigation = {fmt_db(snr_r)} ({arrow} {abs(delta):.1f} dB)"
            )

        QTimer.singleShot(0, update_ui)

    def _on_sim_error(self, message: SimulationErrorMessage):
        def show_error():
            QMessageBox.critical(self, "Simulation Error", message.traceback_text)
            self.shoot_btn.setEnabled(True)
            self.shoot_btn.setText("SHOOT X-RAY")
            self.progress.setVisible(False)
            self.statusBar().showMessage("Simulation failed - see error dialog.")

        QTimer.singleShot(0, show_error)
