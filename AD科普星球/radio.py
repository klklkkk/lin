# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'radio.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QFont, QPixmap

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1184, 927)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.setWindowIcon(QIcon("logo.png"))
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_17 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        spacerItem = QtWidgets.QSpacerItem(996, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_17.addItem(spacerItem)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setStyleSheet("font: 30pt \"华文隶书\"gb(0, 0, 71);")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(2, 1)
        self.verticalLayout_17.addLayout(self.horizontalLayout_3)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_17.addItem(spacerItem3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout.addItem(spacerItem4)
        self.frame1 = QtWidgets.QFrame(self.centralwidget)
        self.frame1.setStyleSheet("border:none")
        self.frame1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame1.setObjectName("frame1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.radio1 = QtWidgets.QPushButton(self.frame1)
        self.radio1.setMinimumSize(QtCore.QSize(300, 150))
        self.radio1.setMaximumSize(QtCore.QSize(300, 16777215))
        self.radio1.setStyleSheet("border-width: 2px;\n"
"border-image: url(images/spjt/1.png);\n"
"border-radius: 10px;\n"
"border-style: outset;\n"
"border-color: rgb(192,192,192);\n"
"background-color:#F5F5F5;")
        self.radio1.setText("")
        self.radio1.setObjectName("radio1")
        self.verticalLayout.addWidget(self.radio1)
        self.label_name = QtWidgets.QLabel(self.frame1)
        self.label_name.setStyleSheet("font: 12pt \"楷体\";\n"
"color: rgb(0, 0, 0);")
        self.label_name.setObjectName("label_name")
        self.verticalLayout.addWidget(self.label_name)
        self.label_source = QtWidgets.QLabel(self.frame1)
        self.label_source.setStyleSheet("font: 10pt \"楷体\";\n"
"color: rgb(168, 168, 168);")
        self.label_source.setObjectName("label_source")
        self.verticalLayout.addWidget(self.label_source)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout.addWidget(self.frame1)
        self.frame2 = QtWidgets.QFrame(self.centralwidget)
        self.frame2.setStyleSheet("border:none")
        self.frame2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame2.setObjectName("frame2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.radio2 = QtWidgets.QPushButton(self.frame2)
        self.radio2.setMinimumSize(QtCore.QSize(300, 150))
        self.radio2.setMaximumSize(QtCore.QSize(300, 16777215))
        self.radio2.setStyleSheet("border-width: 2px;\n"
"border-image: url(images/spjt/2.png);\n"
"border-radius: 10px;\n"
"border-style: outset;\n"
"border-color: rgb(192,192,192);\n"
"background-color:#F5F5F5;")
        self.radio2.setText("")
        self.radio2.setObjectName("radio2")
        self.verticalLayout_4.addWidget(self.radio2)
        self.label_name2 = QtWidgets.QLabel(self.frame2)
        self.label_name2.setStyleSheet("font: 12pt \"楷体\";\n"
"color: rgb(0, 0, 0);")
        self.label_name2.setObjectName("label_name2")
        self.verticalLayout_4.addWidget(self.label_name2)
        self.label_source2 = QtWidgets.QLabel(self.frame2)
        self.label_source2.setStyleSheet("font: 10pt \"楷体\";\n"
"color: rgb(168, 168, 168);")
        self.label_source2.setObjectName("label_source2")
        self.verticalLayout_4.addWidget(self.label_source2)
        self.verticalLayout_4.setStretch(1, 1)
        self.verticalLayout_4.setStretch(2, 1)
        self.verticalLayout_3.addLayout(self.verticalLayout_4)
        self.horizontalLayout.addWidget(self.frame2)
        self.frame3 = QtWidgets.QFrame(self.centralwidget)
        self.frame3.setStyleSheet("border:none")
        self.frame3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame3.setObjectName("frame3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame3)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.radio3 = QtWidgets.QPushButton(self.frame3)
        self.radio3.setMinimumSize(QtCore.QSize(300, 150))
        self.radio3.setMaximumSize(QtCore.QSize(300, 16777215))
        self.radio3.setStyleSheet("border-width: 2px;\n"
"border-image: url(images/spjt/3.png);\n"
"border-radius: 10px;\n"
"border-style: outset;\n"
"border-color: rgb(192,192,192);\n"
"background-color:#F5F5F5;")
        self.radio3.setText("")
        self.radio3.setObjectName("radio3")
        self.verticalLayout_6.addWidget(self.radio3)
        self.label_name3 = QtWidgets.QLabel(self.frame3)
        self.label_name3.setStyleSheet("font: 12pt \"楷体\";\n"
"color: rgb(0, 0, 0);")
        self.label_name3.setObjectName("label_name3")
        self.verticalLayout_6.addWidget(self.label_name3)
        self.label_source3 = QtWidgets.QLabel(self.frame3)
        self.label_source3.setStyleSheet("font: 10pt \"楷体\";\n"
"color: rgb(168, 168, 168);")
        self.label_source3.setObjectName("label_source3")
        self.verticalLayout_6.addWidget(self.label_source3)
        self.verticalLayout_6.setStretch(1, 1)
        self.verticalLayout_6.setStretch(2, 1)
        self.verticalLayout_5.addLayout(self.verticalLayout_6)
        self.horizontalLayout.addWidget(self.frame3)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout.addItem(spacerItem5)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 1)
        self.horizontalLayout.setStretch(3, 1)
        self.verticalLayout_17.addLayout(self.horizontalLayout)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_4.addItem(spacerItem6)
        self.frame4 = QtWidgets.QFrame(self.centralwidget)
        self.frame4.setStyleSheet("border:none")
        self.frame4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame4.setObjectName("frame4")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame4)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.radio4 = QtWidgets.QPushButton(self.frame4)
        self.radio4.setMinimumSize(QtCore.QSize(300, 150))
        self.radio4.setMaximumSize(QtCore.QSize(300, 16777215))
        self.radio4.setStyleSheet("border-width: 2px;\n"
"border-image: url(images/spjt/4.png);\n"
"border-radius: 10px;\n"
"border-style: outset;\n"
"border-color: rgb(192,192,192);\n"
"background-color:#F5F5F5;")
        self.radio4.setText("")
        self.radio4.setObjectName("radio4")
        self.verticalLayout_8.addWidget(self.radio4)
        self.label_name4 = QtWidgets.QLabel(self.frame4)
        self.label_name4.setStyleSheet("font: 12pt \"楷体\";\n"
"color: rgb(0, 0, 0);")
        self.label_name4.setObjectName("label_name4")
        self.verticalLayout_8.addWidget(self.label_name4)
        self.label_source4 = QtWidgets.QLabel(self.frame4)
        self.label_source4.setStyleSheet("font: 10pt \"楷体\";\n"
"color: rgb(168, 168, 168);")
        self.label_source4.setObjectName("label_source4")
        self.verticalLayout_8.addWidget(self.label_source4)
        self.verticalLayout_8.setStretch(1, 1)
        self.verticalLayout_8.setStretch(2, 1)
        self.verticalLayout_7.addLayout(self.verticalLayout_8)
        self.horizontalLayout_4.addWidget(self.frame4)
        self.frame5 = QtWidgets.QFrame(self.centralwidget)
        self.frame5.setStyleSheet("border:none")
        self.frame5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame5.setObjectName("frame5")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.frame5)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.radio5 = QtWidgets.QPushButton(self.frame5)
        self.radio5.setMinimumSize(QtCore.QSize(300, 150))
        self.radio5.setMaximumSize(QtCore.QSize(300, 16777215))
        self.radio5.setStyleSheet("border-width: 2px;\n"
"border-image: url(images/spjt/5.png);\n"
"border-radius: 10px;\n"
"border-style: outset;\n"
"border-color: rgb(192,192,192);\n"
"background-color:#F5F5F5;")
        self.radio5.setText("")
        self.radio5.setObjectName("radio5")
        self.verticalLayout_10.addWidget(self.radio5)
        self.label_name5 = QtWidgets.QLabel(self.frame5)
        self.label_name5.setStyleSheet("font: 12pt \"楷体\";\n"
"color: rgb(0, 0, 0);")
        self.label_name5.setObjectName("label_name5")
        self.verticalLayout_10.addWidget(self.label_name5)
        self.label_source5 = QtWidgets.QLabel(self.frame5)
        self.label_source5.setStyleSheet("font: 10pt \"楷体\";\n"
"color: rgb(168, 168, 168);")
        self.label_source5.setObjectName("label_source5")
        self.verticalLayout_10.addWidget(self.label_source5)
        self.verticalLayout_10.setStretch(1, 1)
        self.verticalLayout_10.setStretch(2, 1)
        self.verticalLayout_9.addLayout(self.verticalLayout_10)
        self.horizontalLayout_4.addWidget(self.frame5)
        self.frame6 = QtWidgets.QFrame(self.centralwidget)
        self.frame6.setStyleSheet("border:none")
        self.frame6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame6.setObjectName("frame6")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.frame6)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.radio6 = QtWidgets.QPushButton(self.frame6)
        self.radio6.setMinimumSize(QtCore.QSize(300, 150))
        self.radio6.setMaximumSize(QtCore.QSize(300, 16777215))
        self.radio6.setStyleSheet("border-width: 2px;\n"
"border-image: url(images/spjt/6.png);\n"
"border-radius: 10px;\n"
"border-style: outset;\n"
"border-color: rgb(192,192,192);\n"
"background-color:#F5F5F5;")
        self.radio6.setText("")
        self.radio6.setObjectName("radio6")
        self.verticalLayout_12.addWidget(self.radio6)
        self.label_name6 = QtWidgets.QLabel(self.frame6)
        self.label_name6.setStyleSheet("font: 12pt \"楷体\";\n"
"color: rgb(0, 0, 0);")
        self.label_name6.setObjectName("label_name6")
        self.verticalLayout_12.addWidget(self.label_name6)
        self.label_source6 = QtWidgets.QLabel(self.frame6)
        self.label_source6.setStyleSheet("font: 10pt \"楷体\";\n"
"color: rgb(168, 168, 168);")
        self.label_source6.setObjectName("label_source6")
        self.verticalLayout_12.addWidget(self.label_source6)
        self.verticalLayout_11.addLayout(self.verticalLayout_12)
        self.horizontalLayout_4.addWidget(self.frame6)
        spacerItem7 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_4.addItem(spacerItem7)
        self.horizontalLayout_4.setStretch(1, 1)
        self.horizontalLayout_4.setStretch(2, 1)
        self.verticalLayout_17.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_2.addItem(spacerItem8)
        self.frame7 = QtWidgets.QFrame(self.centralwidget)
        self.frame7.setStyleSheet("border:none")
        self.frame7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame7.setObjectName("frame7")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.frame7)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.radio7 = QtWidgets.QPushButton(self.frame7)
        self.radio7.setMinimumSize(QtCore.QSize(300, 150))
        self.radio7.setMaximumSize(QtCore.QSize(300, 16777215))
        self.radio7.setStyleSheet("border-width: 2px;\n"
"border-image: url(images/spjt/7.png);\n"
"border-radius: 10px;\n"
"border-style: outset;\n"
"border-color: rgb(192,192,192);\n"
"background-color:#F5F5F5;")
        self.radio7.setText("")
        self.radio7.setObjectName("radio7")
        self.verticalLayout_14.addWidget(self.radio7)
        self.label_name7 = QtWidgets.QLabel(self.frame7)
        self.label_name7.setStyleSheet("font: 12pt \"楷体\";\n"
"color: rgb(0, 0, 0);")
        self.label_name7.setObjectName("label_name7")
        self.verticalLayout_14.addWidget(self.label_name7)
        self.label_source7 = QtWidgets.QLabel(self.frame7)
        self.label_source7.setStyleSheet("font: 10pt \"楷体\";\n"
"color: rgb(168, 168, 168);")
        self.label_source7.setObjectName("label_source7")
        self.verticalLayout_14.addWidget(self.label_source7)
        self.verticalLayout_14.setStretch(1, 1)
        self.verticalLayout_14.setStretch(2, 1)
        self.verticalLayout_13.addLayout(self.verticalLayout_14)
        self.horizontalLayout_2.addWidget(self.frame7)
        self.frame8 = QtWidgets.QFrame(self.centralwidget)
        self.frame8.setStyleSheet("border:none")
        self.frame8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame8.setObjectName("frame8")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.frame8)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout()
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.radio8 = QtWidgets.QPushButton(self.frame8)
        self.radio8.setMinimumSize(QtCore.QSize(300, 150))
        self.radio8.setMaximumSize(QtCore.QSize(300, 16777215))
        self.radio8.setStyleSheet("border-width: 2px;\n"
"border-image: url(images/spjt/8.png);\n"
"border-radius: 10px;\n"
"border-style: outset;\n"
"border-color: rgb(192,192,192);\n"
"background-color:#F5F5F5;")
        self.radio8.setText("")
        self.radio8.setObjectName("radio8")
        self.verticalLayout_16.addWidget(self.radio8)
        self.label_name8 = QtWidgets.QLabel(self.frame8)
        self.label_name8.setStyleSheet("font: 12pt \"楷体\";\n"
"color: rgb(0, 0, 0);")
        self.label_name8.setObjectName("label_name8")
        self.verticalLayout_16.addWidget(self.label_name8)
        self.label_source8 = QtWidgets.QLabel(self.frame8)
        self.label_source8.setStyleSheet("font: 10pt \"楷体\";\n"
"color: rgb(168, 168, 168);")
        self.label_source8.setObjectName("label_source8")
        self.verticalLayout_16.addWidget(self.label_source8)
        self.verticalLayout_16.setStretch(2, 1)
        self.verticalLayout_15.addLayout(self.verticalLayout_16)
        self.horizontalLayout_2.addWidget(self.frame8)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem9)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem10)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem11)
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem12)
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem13)
        spacerItem14 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem14)
        spacerItem15 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem15)
        spacerItem16 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem16)
        spacerItem17 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout_2.addItem(spacerItem17)
        self.horizontalLayout_2.setStretch(1, 1)
        self.verticalLayout_17.addLayout(self.horizontalLayout_2)
        spacerItem18 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_17.addItem(spacerItem18)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "视频播放器"))
        self.label.setText(_translate("MainWindow", "视频播放器"))
        self.label_name.setText(_translate("MainWindow", "阿尔茨海默症患者眼里的世界"))
        self.label_source.setText(_translate("MainWindow", "来源："))
        self.label_name2.setText(_translate("MainWindow", "阿尔茨海默症离我们有多近"))
        self.label_source2.setText(_translate("MainWindow", "来源："))
        self.label_name3.setText(_translate("MainWindow", "阿尔茨海默症无法治愈，你到底能做些什\n"
"么？护理一位患者有多难？"))
        self.label_source3.setText(_translate("MainWindow", "来源："))
        self.label_name4.setText(_translate("MainWindow", "和相爱之人相见却不相识，这种病真的很\n"
"残忍"))
        self.label_source4.setText(_translate("MainWindow", "来源："))
        self.label_name5.setText(_translate("MainWindow", "离谱！学术造假误导人类对阿尔茨海默症\n"
"研究整整16年"))
        self.label_source5.setText(_translate("MainWindow", "来源："))
        self.label_name6.setText(_translate("MainWindow", "如何测试阿尔茨海默症"))
        self.label_source6.setText(_translate("MainWindow", "来源："))
        self.label_name7.setText(_translate("MainWindow", "什么是阿尔茨海默症—法国动画片"))
        self.label_source7.setText(_translate("MainWindow", "来源："))
        self.label_name8.setText(_translate("MainWindow", "失忆、失禁、性情暴躁⋯⋯真实的阿尔茨海\n"
"默症太残酷"))
        self.label_source8.setText(_translate("MainWindow", "来源："))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
