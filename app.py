#!/usr/bin/env python3
"""Entry point for the ELP 3D stereo camera tool."""

import sys

from PyQt5.QtWidgets import QApplication

from elp_stereo.config import load_config
from elp_stereo.gui.main_window import MainWindow


def main():
    cfg = load_config()
    app = QApplication(sys.argv)
    win = MainWindow(cfg)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
