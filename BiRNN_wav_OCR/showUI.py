import sys
import MyUI
from PyQt5.QtWidgets import QApplication, QMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = MyUI.Ui_MyUI()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())