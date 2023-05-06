import sys
import threading
import time

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import QPropertyAnimation, QPoint, QEasingCurve
from PyQt5.QtWidgets import QPushButton, QApplication
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton

from box import Ui_MainWindow

class uic(QMainWindow, Ui_MainWindow):
    conn = pyqtSignal(str)

    def __init__(self):
        super(uic, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("问答机器人")
        self.setWindowIcon(QIcon("logo.png"))
        # self.addToolBar(self.loguobtn)
        self.loadstyle()
    def loadstyle(self):
        self.setStyleSheet('''
                                #centralwidget{
                                    border-image:url(cde.webp);
                                }
                                #frame{
                                    background-color: rgba(255, 255, 225, 5%);
                                }

                                QToolButton{
                                    color:white; 

                                }
                                #textBrowser_2{
                                    background-color: rgba(255,255,255, 50%);
                                }
                                #textBrowser{
                                    background-color: rgba(255,255,255, 50%);
                                }
                                #plainTextEdit{
                                    background-color: rgba(255,255,255, 75%);
                                }
                                QLabel{
                                    background-color: transparent;

                                    color:black;
                                }
                                #frame_5,#frame_6,#frame_7,#frame_8,#frame_9,#frame_10,#frame_11,#frame_12,#frame_13,#frame_14{
                                    background-color: rgba(0, 0, 0, 10%);
                                }

                                ''')
        self.pushButton.setIcon(QIcon("1.png"))
        self.pushButton.setIconSize(QSize(130, 68))
        self.pushButton.setObjectName("btn2")


        self.logolb.setPixmap(QPixmap("2.png"))
        self.logolb.setScaledContents(True)
        #self.loguobtn.setIcon(QIcon("out.png"))
        # self.logolb.setPixmap(QPixmap("1.png"))
        # self.logolb.setScaledContents(True)
        # self.btn1.setStyleSheet("QToolButton{background-color: rgba(0, 0, 0, 10%);}QToolButton:hover{background-color: rgba(255, 255, 255, 10%);border:1px solid white}")

class dialog:
    def __init__(self):
        self.ui = uic()
        self.ui.pushButton.clicked.connect(self.on_button_clicked)

        self.brs = self.ui.textBrowser
        self.css = "<style>" + \
                ".atalk span {" + \
                    "display: inline-block;" + \
                    "border-radius: 10px;" + \
                    "padding: 5px 10px;" + \
                    "color: #000000;" + \
                "}" + \
                ".btalk span {" + \
                    "display: inline-block;" + \
                    "background-color: #4094C9;" + \
                    "border-radius: 10px;" + \
                    "padding: 5px 10px;" + \
                    "color: #ffffff;" + \
                "}" + \
                "</style>"
        self.html = "<html><head></head>" + self.css + \
               "<body>"
        self.tail = "</body></html>"
        self.brs.setHtml(self.html)

    def on_button_clicked(self):
        query = self.ui.plainTextEdit.toPlainText()
        self.ui.plainTextEdit.setPlainText("")
        self.brs.append(self.html+"<div class='atalk'><span style='font-size:25px;'>Q：" + query + "</span></div>"+ self.tail)
        answer = "抱歉，没有找到您所需要的答案"
        with open("Q&A.txt", "r", encoding="gbk") as file:
            content = file.read()
            lines = content.split("$\n")
            for i in range(0, len(lines), 2):
                for l in range(len(query)):
                    for r in range(l, len(query)):
                        if query[l:r+1] == lines[i]:
                            answer = lines[i+1]
        self.brs.append(self.html+"<div class='btalk'><span style='font-size:25px;'>A：" + answer + "</span></div>" + self.tail)
        self.brs.append(self.html + "<div class='btalk'></div>" + self.tail)
        # print(self.brs.toHtml())
