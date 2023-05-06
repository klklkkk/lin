import sys
from articlesearchfunction import ArticleSearchFunction
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, QSize, pyqtSlot
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QHeaderView, QTableWidgetItem, QCheckBox,QSystemTrayIcon
from ui import Ui_MainWindow
from rummormillfunction import RummorMillFunction
from run_radio import winradio
from robot import dialog
from Dynamicdisplayfunction import DynamicDisplayFunction
from zctrler import ctrl
import ctypes
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")
class uic(QMainWindow, Ui_MainWindow):
    conn = pyqtSignal(str)
    def __init__(self):
        super(uic, self).__init__()
        self.rumor_mill = None
        self.article_search = None
        self.setupUi(self)
        self.setWindowTitle("阿尔兹海默科普星球")
        self.setWindowIcon(QIcon('logo.png'))
        self.loadstyle()

    def loadstyle(self):
        self.setStyleSheet('''
                                #centralwidget{
                                    border-image:url(bg3.jpg);
                                }
                                #frame{
                                    background-color: rgba(255, 255, 225, 5%);
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
        self.logolb.setPixmap(QPixmap("nao.png"))
        self.logolb.setScaledContents(True)
        #self.btn1.setStyleSheet("QToolButton{background-color: rgba(0, 0, 0, 10%);}QToolButton:hover{background-color: rgba(255, 255, 255, 10%);border:1px solid white}")
        for i in range(0, 6):
            s = "self.btn{}.setIconSize(QSize(206,93))".format(str(i + 1))
            eval(s)
            s = "self.btn{}.setIcon(QIcon('img/{}'))".format(str(i+1), "img_"+str(i)+".png")
            eval(s)


    # 主页点击文章搜索按钮
    @pyqtSlot()
    def on_btn6_clicked(self):
        self.article_search = ArticleSearchFunction()
        self.article_search.show()

    # 主页点击分类预测按钮
    @pyqtSlot()
    def on_btn1_clicked(self):
        self.predict = ctrl()
        self.predict.ui.show()
    # 主页点击谣言粉碎机按钮
    @pyqtSlot()
    def on_btn2_clicked(self):
        self.rumor_mill = RummorMillFunction()
        #self.rumor_mill.setFixedSize(1200, 600)
        self.rumor_mill.show()

    # 主页点击谣言粉碎机按钮
    @pyqtSlot()
    def on_btn3_clicked(self):
        self.run_radio = winradio()
        self.run_radio.show()

    # 主页点击问答机器人
    @pyqtSlot()
    def on_btn4_clicked(self):
        self.run_robot = dialog()
        self.run_robot.ui.show()

    # 主页点击动态展示按钮
    @pyqtSlot()
    def on_btn5_clicked(self):
        self.dynamic_display = DynamicDisplayFunction()
        self.dynamic_display.show()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = uic()

    ui.show()
    sys.exit(app.exec_())