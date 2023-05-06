# -*- coding: utf-8 -*-
import radio
from PyQt5 import QtWidgets
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtMultimediaWidgets import QVideoWidget
radioname=''
class winradioshow(QMainWindow):
    def __init__(self):
        super(winradioshow, self).__init__()
        self.setupUi(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 750)
        MainWindow.setStyleSheet("    background-color:#ffffff;\n")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.wgt_video = QVideoWidget(self.centralwidget)  # 创建一个 视频播放组件
        self.wgt_video.setGeometry(QtCore.QRect(100, 80, 800, 500))
        self.wgt_video.setObjectName("wgt_video")
        self.btn_open = QtWidgets.QPushButton(self.centralwidget)
        self.btn_open.setGeometry(QtCore.QRect(415, 600, 120, 50))
        self.btn_open.setObjectName("btn_open")
        self.btn_open.setStyleSheet("QPushButton{\n"
"    background-color:#4e6ef2;\n"
"    color:rgb(255, 255, 255);\n"
"    border:2px solid #4e6ef2;\n"
"    border-radius:5px;\n"
"    \n"
"    font: 12pt \"楷体\";\n"
"}\n"
"QPushButton:pressed{\n"
"    padding-top:5px;\n"
"    padding-left:5px;\n"
"}")
        MainWindow.setCentralWidget(self.centralwidget)
        # 获取屏幕坐标系
        screen = QDesktopWidget().screenGeometry()
        # 获取窗口坐标系
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft), int(newTop)+50)
        #隐蔽窗口
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.retranslateUi(MainWindow)
        self.openVideoFile()
        self.btn_open.clicked.connect(self.winquit)   # 打开视频文件按钮
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    #实现窗口拖动
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, mouse_event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(mouse_event.globalPos() - self.m_Position)  # 更改窗口位置
            mouse_event.accept()

    def mouseReleaseEvent(self, mouse_event):
        self.m_flag = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
    # 窗口拖动结束
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "视频播放"))
        self.btn_open.setText(_translate("MainWindow", "退出"))

        self.player = QMediaPlayer()  # 创建视频播放管理器
        self.player.setVideoOutput(self.wgt_video)  # 视频播放输出的widget，就是上面定义的
        
    def openVideoFile(self):
        global radioname
        #print('open file')
        #选择本地视频播放
         # 播放网络视频
        #self.player.setMedia(QMediaContent(QtCore.QUrl.fromLocalFile("https://vd1.bdstatic.com/mda-hg6uempmez9u6mqi/sc/mda-hg6uempmez9u6mqi.mp4?auth_key=1562172911-0-0-4c22196ad1d0fcc49402d91336c999c5&bcevod_channel=searchbox_feed&pd=bjh&abtest=all")))
        self.player.setMedia(QMediaContent(QtCore.QUrl.fromLocalFile(f"sp/{radioname}.mp4")))  # 选取本地视频文件
        self.player.play()  # 播放视频
    def winquit(self):
        self.player.stop()
        self.close()
class winradio(radio.Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(winradio, self).__init__()
        self.setupUi(self)
        # 获取屏幕坐标系
        screen = QDesktopWidget().screenGeometry()
        # 获取窗口坐标系
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft), int(newTop))
        self.radio1.clicked.connect(self.radio1_show)  # 将信号连接到槽
        self.radio2.clicked.connect(self.radio2_show)  # 将信号连接到槽
        self.radio3.clicked.connect(self.radio3_show)  # 将信号连接到槽
        self.radio4.clicked.connect(self.radio4_show)  # 将信号连接到槽
        self.radio5.clicked.connect(self.radio5_show)  # 将信号连接到槽
        self.radio6.clicked.connect(self.radio6_show)  # 将信号连接到槽
        self.radio7.clicked.connect(self.radio7_show)  # 将信号连接到槽
        self.radio8.clicked.connect(self.radio8_show)  # 将信号连接到槽
    def radio1_show(self):
        global radioname
        radioname='1'
        self.winshow = winradioshow()  # 将第一和窗口换个名字
        # 创建窗口
        self.winshow.show()  # 将第一和窗口换个名字显示出
    def radio2_show(self):
        global radioname
        radioname='2'
        self.winshow = winradioshow()  # 将第一和窗口换个名字
        # 创建窗口
        self.winshow.show()  # 将第一和窗口换个名字显示出
    def radio3_show(self):
        global radioname
        radioname='3'
        self.winshow = winradioshow()  # 将第一和窗口换个名字
        # 创建窗口
        self.winshow.show()  # 将第一和窗口换个名字显示出
    def radio4_show(self):
        global radioname
        radioname='4'
        self.winshow = winradioshow()  # 将第一和窗口换个名字
        # 创建窗口
        self.winshow.show()  # 将第一和窗口换个名字显示出
    def radio5_show(self):
        global radioname
        radioname='5'
        self.winshow = winradioshow()  # 将第一和窗口换个名字
        # 创建窗口
        self.winshow.show()  # 将第一和窗口换个名字显示出
    def radio6_show(self):
        global radioname
        radioname='6'
        self.winshow = winradioshow()  # 将第一和窗口换个名字
        # 创建窗口
        self.winshow.show()  # 将第一和窗口换个名字显示出
    def radio7_show(self):
        global radioname
        radioname='7'
        self.winshow = winradioshow()  # 将第一和窗口换个名字
        # 创建窗口
        self.winshow.show()  # 将第一和窗口换个名字显示出
    def radio8_show(self):
        global radioname
        radioname='8'
        self.winshow = winradioshow()  # 将第一和窗口换个名字
        # 创建窗口
        self.winshow.show()  # 将第一和窗口换个名字显示出
def run():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    radioUi = winradio()  # 将第一和窗口换个名字
    # 创建窗口
    radioUi.show()  # 将第一和窗口换个名字显示出
    sys.exit(app.exec_())  # app.exet_()是指程序一直循环运行直到主窗口被关闭终止进程（如果没有这句话，程序运行时会一闪而过）




