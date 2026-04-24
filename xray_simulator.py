#!/usr/bin/env python3
"""Entry point for the modular X-ray motion artifact simulator."""

import sys
import warnings

import matplotlib
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

from imaging_app.main_window import MainWindow

warnings.filterwarnings("ignore")
matplotlib.use("Qt5Agg")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#0e0e1c"))
    pal.setColor(QPalette.WindowText, QColor("#c8cce8"))
    pal.setColor(QPalette.Base, QColor("#161628"))
    pal.setColor(QPalette.AlternateBase, QColor("#0e0e1c"))
    pal.setColor(QPalette.ToolTipBase, QColor("#161628"))
    pal.setColor(QPalette.ToolTipText, QColor("#c8cce8"))
    pal.setColor(QPalette.Text, QColor("#c8cce8"))
    pal.setColor(QPalette.Button, QColor("#161628"))
    pal.setColor(QPalette.ButtonText, QColor("#c8cce8"))
    pal.setColor(QPalette.BrightText, QColor("#ffffff"))
    pal.setColor(QPalette.Highlight, QColor("#2a4080"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
