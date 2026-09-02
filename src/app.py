from __future__ import annotations

import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from src.ui.main_window import MainWindow


def install_excepthook():
    def hook(exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(text, file=sys.stderr)
        try:
            QMessageBox.critical(
                None, "予期しないエラー",
                f"{exc_type.__name__}: {exc_value}\n\n詳細はターミナルログを確認してください。",
            )
        except Exception:
            pass
    sys.excepthook = hook


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI物体検出結果レビューツール")
    install_excepthook()
    win = MainWindow()
    win.resize(1920, 1080)
    win.showMaximized()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
