import sys
import threading
import time
import os
import shutil
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import QMainWindow, QApplication, QHeaderView, QTableWidgetItem, QCheckBox, QFileDialog, QMessageBox

from box2 import Ui_MainWindow
from pcmine import predict


class uic(QMainWindow, Ui_MainWindow):
    conn = pyqtSignal(str)

    def __init__(self):
        super(uic, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("预测功能")
        self.setWindowIcon(QIcon("logo.png"))
        self.loguobtn.clicked.connect(self.close)
        # self.addToolBar(self.loguobtn)
        self.loadstyle()
        self.progressBar.setValue(0)
        self.progressBar2.setValue(0)
        self.progressBar3.setValue(0)

    def loadstyle(self):
        self.setStyleSheet('''
                                #centralwidget{
                                    border-image:url(bg.jfif);
                                }
                                #frame{
                                    background-color: rgba(255, 255, 225, 5%);
                                }
                                QRadioButton{
                                    color:white; 

                                }
                                QToolButton{
                                    color:white; 

                                }
                                QLabel{
                                    background-color: transparent;

                                    color:white;
                                }
                                #frame_5,#frame_6,#frame_7,#frame_8,#frame_9,#frame_10,#frame_11,#frame_12,#frame_13,#frame_14{
                                    background-color: rgba(0, 0, 0, 10%);
                                }

                                ''')

        self.loguobtn.setIcon(QIcon("out.png"))
        # self.logolb.setPixmap(QPixmap("1.png"))
        # self.logolb.setScaledContents(True)
        # self.btn1.setStyleSheet("QToolButton{background-color: rgba(0, 0, 0, 10%);}QToolButton:hover{background-color: rgba(255, 255, 255, 10%);border:1px solid white}")


class ctrl:
    def __init__(self):
        self.ui = uic()
        # 调整最下方的字体
        font = QFont()
        font.setFamily("Arial")
        font.setBold(True)


        '''
        palette = QPalette()
        palette.setColor(QPalette.Foreground, Qt.white)

        '''

        # 放入图片
        pixmap = QPixmap("PR.png")
        # 将QPixmap对象设置为QLabel控件的图像
        self.ui.label_12.setPixmap(pixmap)
        # 调整QLabel控件的大小以适应图像大小
        self.ui.label_12.setScaledContents(True)
        pixmap = QPixmap("Mtrx.png")
        # 将QPixmap对象设置为QLabel控件的图像
        self.ui.label_13.setPixmap(pixmap)
        # 调整QLabel控件的大小以适应图像大小
        self.ui.label_13.setScaledContents(True)
        pixmap = QPixmap("ROC.png")
        # 将QPixmap对象设置为QLabel控件的图像
        self.ui.label_14.setPixmap(pixmap)
        # 调整QLabel控件的大小以适应图像大小
        self.ui.label_14.setScaledContents(True)



        # 逻辑控件
        self.ui.matButton.clicked.connect(self.on_matButton_clicked)
        self.ui.txtButton.clicked.connect(self.on_txtButton_clicked)
        self.ui.pushButton.clicked.connect(self.on_pushButton_clicked)
        self.has_mat = 0
        self.has_txt = 0
    def dlt(self): # 删除两个文件夹中的所有文件
        folder_path = 'user_feature'  # 文件夹路径
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        folder_path = 'userB'  # 文件夹路径
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("Delete!")
    def on_matButton_clicked(self):
        file_dialog = QFileDialog()

        # set the dialog options (optional)
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setNameFilter("Text files (*.txt)")

        # open the file dialog and retrieve the selected file's path
        file_path, _ = file_dialog.getOpenFileName(None, "选择一个.mat文件上传", "", "All files (*.mat)")
        # file_path, _ = QFileDialog.getOpenFileName(self, "选择一个.mat文件上传", ".", "All Files (*)")
        if file_path:
            # 指定保存文件的目录
            save_dir = 'userB'
            # 从文件路径中获取文件名
            file_name = os.path.basename(file_path)
            # 拼接保存文件的完整路径
            save_path = os.path.join(save_dir, file_name)
            # 复制文件到指定目录中
            shutil.copyfile(file_path, save_path)
            self.has_mat = 1
            self.progressBar = self.ui.progressBar
            self.set_loader()
            print(".mat上传成功！")

    def on_txtButton_clicked(self):

        file_dialog = QFileDialog()

        # set the dialog options (optional)
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setNameFilter("Text files (*.txt)")

        # open the file dialog and retrieve the selected file's path
        file_path, _ = file_dialog.getOpenFileName(None, "选择一个.txt文件上传", "", "All files (*.txt)")
        # file_path, _ = QFileDialog.getOpenFileName(self, "选择一个.txt文件上传", ".", "All Files (*)")
        if file_path:
            # 指定保存文件的目录
            save_dir = 'user_feature'
            # 从文件路径中获取文件名
            file_name = os.path.basename(file_path)
            # 拼接保存文件的完整路径
            save_path = os.path.join(save_dir, file_name)
            # 复制文件到指定目录中
            shutil.copyfile(file_path, save_path)
            self.has_txt = 1
            self.progressBar = self.ui.progressBar2
            self.set_loader()
            print(".txt上传成功！")

    def on_pushButton_clicked(self):
        if self.has_mat and self.has_txt:
            pd = predict()
            if self.ui.radioButton.isChecked() == 1:
                res = pd.work_with_grandnorm().item()
            else:
                res = pd.work().item()
            print(res)
            self.progressBar = self.ui.progressBar3
            self.set_loader()
            if res == 0:
                self.ui.answerButton.setText("正常人")
            else:
                self.ui.answerButton.setText("病人")
        elif self.has_mat == 1 and self.has_txt == 0:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setWindowIcon(QIcon("logo.png"))
            msg_box.setText("请先上传.txt文件再启动预测")
            msg_box.exec_()
        elif self.has_mat == 0 and self.has_txt == 1:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setWindowIcon(QIcon("logo.png"))
            msg_box.setText("请先上传.mat文件再启动预测")
            msg_box.exec_()
        else:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setWindowIcon(QIcon("logo.png"))
            msg_box.setText("请先上传.mat文件和.txt文件再启动预测")
            msg_box.exec_()
        self.dlt()
        self.has_mat = 0
        self.has_txt = 0

    def set_loader(self):
        self.progressBar.setValue(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.load_progress_bar)
        self.timer.start(3)

    def load_progress_bar(self):
        self.progressBar.setValue(self.progressBar.value() + 1)
        if self.progressBar.value() >= 100:
            self.timer.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    c = ctrl()
    c.ui.show()
    sys.exit(app.exec_())