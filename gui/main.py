from PySide6.QtWidgets import QApplication

import sys
import argparse

from widgets import MainWindow


def main():
    app = QApplication(sys.argv)
    
    window = MainWindow(app=app, )
    window.show()

    app.exec()

if __name__ == "__main__":
    main()