import sys
from PyQt5.QtWidgets import QApplication

from utility.yolodetectiongui import YOLODetectionGUI

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = YOLODetectionGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()