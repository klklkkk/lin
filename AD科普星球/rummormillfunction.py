from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget
from rumor_mill import Ui_Form
import pandas as pd
import random
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5 import QtCore, QtGui, QtWidgets
str_num=6
show_str = 22
class RummorMillFunction(QWidget):
    """
    实现谣言粉碎机功能
    """

    def __init__(self):
        super(RummorMillFunction, self).__init__()
        self.Ui = Ui_Form()  # 实例化 Ui
        self.Ui.setupUi(self)  # 初始化Ui
        self.Ui.retranslateUi(self)
        self.setWindowTitle('谣言粉碎机')
        self.setFixedSize(self.width(), self.height())
        self.btn_move()
        self.setWindowIcon(QIcon("logo.png"))
        self.btn_old_height = None
        self.btn_old_width = None
        self.click_state = 0
        self.btn_num = 7
        self.index = []
        self.btn_state = ['T', 'T', 'T', 'T', 'T', 'T', 'T']
        self.data = pd.read_excel('data/rumor_mill.xlsx', sheet_name=0)
        self.show_data()


    def btn_move(self):
        screen = QDesktopWidget().screenGeometry()
        y1 = random.randint(10, int(screen.height())-600)
        y2 = random.randint(10, int(screen.height())-600)
        y3 = random.randint(10, int(screen.height())-600)
        y4 = random.randint(10, int(screen.height())-600)
        y5 = random.randint(10, int(screen.height())-600)
        y6 = random.randint(10, int(screen.height())-600)
        y7 = random.randint(10, int(screen.height())-600)
        x1 = random.randint(20, 100)
        x2 = random.randint(120, 250)
        x3 = random.randint(270, 370)
        x4 = random.randint(390, 500)
        x5 = random.randint(520, 630)
        x6 = random.randint(650, 760)
        x7 = random.randint(780, 900)
        self.Ui.btn1.move(x1, y1)
        self.Ui.btn2.move(x2, y2)
        self.Ui.btn3.move(x3, y3)
        self.Ui.btn4.move(x4, y4)
        self.Ui.btn5.move(x5, y5)
        self.Ui.btn6.move(x6, y6)
        self.Ui.btn7.move(x7, y7)
    def show_data(self):
        str=[]
        # 随机生成索引
        self.index = random.sample(range(0, int(self.data.shape[0])), self.btn_num)
        for num in range(7):
            str.append(list(self.data.at[self.index[num], "Q"]))
        for _n in range(7):
            cishu = len(str[_n]) / 7
            for i2 in range(int(cishu) + 1):
                str[_n].insert(7 * i2, '\n')
            str[_n] = ''.join(str[_n])
        self.Ui.btn1.setText(str[0])
        self.Ui.btn2.setText(str[1])
        self.Ui.btn3.setText(str[2])
        self.Ui.btn4.setText(str[3])
        self.Ui.btn5.setText(str[4])
        self.Ui.btn6.setText(str[5])
        self.Ui.btn7.setText(str[6])

    @pyqtSlot()
    def on_refresh_btn_clicked(self):
        self.btn_move()
        self.show_data()

    @pyqtSlot()
    def on_btn1_clicked(self):
        self.click_state = 1 ^ self.click_state
        size = 600 + self.click_state * 600
        if self.click_state:
            self.btn_old_width = self.Ui.btn1.geometry().x()
            self.btn_old_height = self.Ui.btn1.geometry().y()
            self.Ui.btn1.setGeometry(int(self.width() / 2) - 600, int(self.height() / 2) - 600, size, size)
            self.set_but_state(0, abled=False)
            text = list(self.data.at[self.index[0], "A"])
            cishu = len(text) / show_str
            for i in range(int(cishu) + 1):
                text.insert(show_str * i, '\n')
            text = ''.join(text)
            self.Ui.btn1.setText(text)
            self.Ui.btn1.setStyleSheet("border-image: url(img/water.png);\n"
                                       "background-color: rgba(0,0, 0, 0);\n"
                                       "")
            # print(self.data.at[self.index[0], "A"])
        else:
            size = 200 + self.click_state * 200
            self.Ui.btn1.setGeometry(self.btn_old_width, self.btn_old_height, size, size)
            self.set_but_state(abled=True)
            str = list(self.data.at[self.index[0], "Q"])
            cishu = len(str) / str_num
            for i in range(int(cishu) + 1):
                str.insert(str_num * i, '\n')
            str = ''.join(str)
            self.Ui.btn1.setText(str)
            self.Ui.btn1.setStyleSheet("border-image: url(img/qipao.png);\n"
                                    "background-color: rgba(0,0, 0, 0);\n"
                                    "")
            self.btn_move()
        self.Ui.btn1.setFixedSize(size, size)
        self.Ui.btn1.raise_()

    @pyqtSlot()
    def on_btn2_clicked(self):
        self.click_state = 1 ^ self.click_state
        size = 600 + self.click_state * 600
        if self.click_state:
            self.btn_old_width = self.Ui.btn2.geometry().x()
            self.btn_old_height = self.Ui.btn2.geometry().y()
            self.Ui.btn2.setGeometry(int(self.width() / 2) - 600, int(self.height() / 2) - 600, size, size)
            self.set_but_state(1, abled=False)
            text = list(self.data.at[self.index[1], "A"])
            cishu = len(text) / show_str
            for i in range(int(cishu) + 1):
                text.insert(show_str * i, '\n')
            text = ''.join(text)
            self.Ui.btn2.setText(text)
            self.Ui.btn2.setStyleSheet("border-image: url(img/water.png);\n"
                                    "background-color: rgba(0,0, 0, 0);\n"
                                    "")
        else:
            size = 200 + self.click_state * 200
            self.Ui.btn2.setGeometry(self.btn_old_width, self.btn_old_height, size, size)
            self.set_but_state(abled=True)
            str = list(self.data.at[self.index[1], "Q"])
            cishu = len(str) / str_num
            for i in range(int(cishu) + 1):
                str.insert(str_num * i, '\n')
            str = ''.join(str)
            self.Ui.btn2.setText(str)
            self.Ui.btn2.setStyleSheet("border-image: url(img/qipao.png);\n"
                                    "background-color: rgba(0,0, 0, 0);\n"
                                    "")
            self.btn_move()
        self.Ui.btn2.setFixedSize(size, size)
        self.Ui.btn2.raise_()

    @pyqtSlot()
    def on_btn3_clicked(self):
        self.click_state = 1 ^ self.click_state
        size = 600 + self.click_state * 600
        if self.click_state:
            self.btn_old_width = self.Ui.btn3.geometry().x()
            self.btn_old_height = self.Ui.btn3.geometry().y()
            self.Ui.btn3.setGeometry(int(self.width() / 2) - 600, int(self.height() / 2) - 600, size, size)
            self.set_but_state(2, abled=False)
            text = list(self.data.at[self.index[2], "A"])
            cishu = len(text) / show_str
            for i in range(int(cishu) + 1):
                text.insert(show_str * i, '\n')
            text = ''.join(text)
            self.Ui.btn3.setText(text)
            self.Ui.btn3.setStyleSheet("border-image: url(img/water.png);\n"
                                    "background-color: rgba(0,0, 0, 0);\n"
                                    "")
        else:
            size = 200 + self.click_state * 200
            self.Ui.btn3.setGeometry(self.btn_old_width, self.btn_old_height, size, size)
            self.set_but_state(abled=True)
            str = list(self.data.at[self.index[2], "Q"])
            cishu = len(str) / str_num
            for i in range(int(cishu) + 1):
                str.insert(str_num * i, '\n')
            str = ''.join(str)
            self.Ui.btn3.setText(str)
            self.Ui.btn3.setStyleSheet("border-image: url(img/qipao.png);\n"
                                    "background-color: rgba(0,0, 0, 0);\n"
                                    "")
            self.btn_move()
        self.Ui.btn3.setFixedSize(size, size)
        self.Ui.btn3.raise_()

    @pyqtSlot()
    def on_btn4_clicked(self):
        self.click_state = 1 ^ self.click_state
        size = 600 + self.click_state * 600
        if self.click_state:
            self.btn_old_width = self.Ui.btn4.geometry().x()
            self.btn_old_height = self.Ui.btn4.geometry().y()
            self.Ui.btn4.setGeometry(int(self.width() / 2) - 600, int(self.height() / 2) - 600, size, size)
            self.set_but_state(3, abled=False)
            text = list(self.data.at[self.index[3], "A"])
            cishu = len(text) / show_str
            for i in range(int(cishu) + 1):
                text.insert(show_str * i, '\n')
            text = ''.join(text)
            self.Ui.btn4.setText(text)
            self.Ui.btn4.setStyleSheet("border-image: url(img/water.png);\n"
                                    "background-color: rgba(0,0, 0, 0);\n"
                                    "")
        else:
            size = 200 + self.click_state * 200
            self.Ui.btn4.setGeometry(self.btn_old_width, self.btn_old_height, size, size)
            self.set_but_state(abled=True)
            str = list(self.data.at[self.index[3], "Q"])
            cishu = len(str) / str_num
            for i in range(int(cishu) + 1):
                str.insert(str_num * i, '\n')
            str = ''.join(str)
            self.Ui.btn4.setText(str)
            self.Ui.btn4.setStyleSheet("border-image: url(img/qipao.png);\n"
                                    "background-color: rgba(0,0, 0, 0);\n"
                                    "")
            self.btn_move()
        self.Ui.btn4.setFixedSize(size, size)
        self.Ui.btn4.raise_()


    @pyqtSlot()
    def on_btn5_clicked(self):
        self.click_state = 1 ^ self.click_state
        size = 600 + self.click_state * 600
        if self.click_state:
            self.btn_old_width = self.Ui.btn5.geometry().x()
            self.btn_old_height = self.Ui.btn5.geometry().y()
            self.Ui.btn5.setGeometry(int(self.width() / 2) - 600, int(self.height() / 2) - 600, size, size)
            self.set_but_state(4, abled=False)
            text = list(self.data.at[self.index[4], "A"])
            cishu = len(text) / show_str
            for i in range(int(cishu) + 1):
                text.insert(show_str * i, '\n')
            text = ''.join(text)
            self.Ui.btn5.setText(text)
            self.Ui.btn5.setStyleSheet("border-image: url(img/water.png);\n"
                                    "background-color: rgba(0,0, 0, 0);\n"
                                    "")
        else:
            size = 200 + self.click_state * 200
            self.Ui.btn5.setGeometry(self.btn_old_width, self.btn_old_height, size, size)
            self.set_but_state(abled=True)
            str = list(self.data.at[self.index[4], "Q"])
            cishu = len(str) / str_num
            for i in range(int(cishu) + 1):
                str.insert(str_num * i, '\n')
            str = ''.join(str)
            self.Ui.btn5.setText(str)
            self.Ui.btn5.setStyleSheet("border-image: url(img/qipao.png);\n"
                                       "background-color: rgba(0,0, 0, 0);\n"
                                       "")
            self.btn_move()
        self.Ui.btn5.setFixedSize(size, size)
        self.Ui.btn5.raise_()


    @pyqtSlot()
    def on_btn6_clicked(self):
        self.click_state = 1 ^ self.click_state
        size = 600 + self.click_state * 600
        if self.click_state:
            self.btn_old_width = self.Ui.btn6.geometry().x()
            self.btn_old_height = self.Ui.btn6.geometry().y()
            self.Ui.btn6.setGeometry(int(self.width() / 2) - 600, int(self.height() / 2) - 600, size, size)
            self.set_but_state(5, abled=False)
            text = list(self.data.at[self.index[5], "A"])
            cishu = len(text) / show_str
            for i in range(int(cishu) + 1):
                text.insert(show_str * i, '\n')
            text = ''.join(text)
            self.Ui.btn6.setText(text)
            self.Ui.btn6.setStyleSheet("border-image: url(img/water.png);\n"
                                    "background-color: rgba(0,0, 0, 0);\n"
                                    "")
        else:
            size = 200 + self.click_state * 200
            self.Ui.btn6.setGeometry(self.btn_old_width, self.btn_old_height, size, size)
            self.set_but_state(abled=True)
            str = list(self.data.at[self.index[5], "Q"])
            cishu = len(str) / str_num
            for i in range(int(cishu) + 1):
                str.insert(str_num * i, '\n')
            str = ''.join(str)
            self.Ui.btn6.setText(str)
            self.Ui.btn6.setStyleSheet("border-image: url(img/qipao.png);\n"
                                       "background-color: rgba(0,0, 0, 0);\n"
                                       "")
            self.btn_move()
        self.Ui.btn6.setFixedSize(size, size)
        self.Ui.btn6.raise_()

    @pyqtSlot()
    def on_btn7_clicked(self):
        self.click_state = 1 ^ self.click_state
        size = 600 + self.click_state * 600
        if self.click_state:
            self.btn_old_width = self.Ui.btn7.geometry().x()
            self.btn_old_height = self.Ui.btn7.geometry().y()
            self.Ui.btn7.setGeometry(int(self.width() / 2) - 600, int(self.height() / 2) - 600, size, size)
            self.set_but_state(6, abled=False)
            text = list(self.data.at[self.index[6], "A"])
            cishu = len(text) / show_str
            for i in range(int(cishu) + 1):
                text.insert(show_str * i, '\n')
            text = ''.join(text)
            self.Ui.btn7.setText(text)
            self.Ui.btn7.setStyleSheet("border-image: url(img/water.png);\n"
                                    "background-color: rgba(0,0, 0, 0);\n"
                                    "")
        else:
            size = 200 + self.click_state * 200
            self.Ui.btn7.setGeometry(self.btn_old_width, self.btn_old_height, size, size)
            self.set_but_state(abled=True)
            str = list(self.data.at[self.index[6], "Q"])
            cishu = len(str) / str_num
            for i in range(int(cishu) + 1):
                str.insert(str_num * i, '\n')
            str = ''.join(str)
            self.Ui.btn7.setText(str)
            self.Ui.btn7.setStyleSheet("border-image: url(img/qipao.png);\n"
                                       "background-color: rgba(0,0, 0, 0);\n"
                                       "")
            self.btn_move()
        self.Ui.btn7.setFixedSize(size, size)
        self.Ui.btn7.raise_()

    def set_but_state(self, i=0, abled=True):
        if abled:
            for btn in range(len(self.btn_state)):
                self.btn_state[btn] = 'T'
        else:
            for btn in range(len(self.btn_state)):
                self.btn_state[btn] = 'F'
            self.btn_state[i] = 'T'
        if self.btn_state[0] == 'F':
            self.Ui.btn1.setEnabled(False)
        else:
            self.Ui.btn1.setEnabled(True)
        if self.btn_state[1] == 'F':
            self.Ui.btn2.setEnabled(False)
        else:
            self.Ui.btn2.setEnabled(True)
        if self.btn_state[2] == 'F':
            self.Ui.btn3.setEnabled(False)
        else:
            self.Ui.btn3.setEnabled(True)
        if self.btn_state[3] == 'F':
            self.Ui.btn4.setEnabled(False)
        else:
            self.Ui.btn4.setEnabled(True)
        if self.btn_state[4] == 'F':
            self.Ui.btn5.setEnabled(False)
        else:
            self.Ui.btn5.setEnabled(True)
        if self.btn_state[5] == 'F':
            self.Ui.btn6.setEnabled(False)
        else:
            self.Ui.btn6.setEnabled(True)
        if self.btn_state[6] == 'F':
            self.Ui.btn7.setEnabled(False)
        else:
            self.Ui.btn7.setEnabled(True)

if __name__ == "__main__":
    import sys
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = RummorMillFunction()
    MainWindow.showMaximized()
    sys.exit(app.exec_())