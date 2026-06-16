"""Application constants and styling."""

VOXEL_SIZE = 0.30
DEFAULT_SOD_CM = 60.0
DEFAULT_SDD_CM = 120.0

MU = dict(
    air=0.000,
    fat=0.160,
    soft_tissue=0.200,
    muscle=0.210,
    blood=0.222,
    liver=0.215,
    lung=0.045,
    cortical=0.520,
    spongy_bone=0.370,
)

BODY_PARTS = {
    "Full Body": (-1.00, 1.00),
    "Head": (0.58, 1.00),
    "Chest": (0.00, 0.57),
    "Abdomen": (-0.42, 0.05),
    "Pelvis": (-0.92, -0.38),
}

PROJ_AXES = {
    "AP  (Front -> Back)": 1,
    "PA  (Back -> Front)": 1,
    "Lateral  (Left -> Right)": 0,
    "Lateral  (Right -> Left)": 0,
}

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
