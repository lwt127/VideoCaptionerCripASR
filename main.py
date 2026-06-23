"""VideoCaptioner application entry point.

Run with:  python main.py

Requires the dependencies in requirements.txt and (for transcription) the
CrispASR / Faster-Whisper backends described in the README.
"""

import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication


def main() -> int:
    # Enable High-DPI scaling for crisp UI on high-resolution displays.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)

    # Import after QApplication so Qt resources/config initialise correctly.
    from app.view.main_window import MainWindow

    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    # Make sure the project root is importable as the `app` package.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
