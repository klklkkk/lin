from PyQt5.QtCore import QPropertyAnimation, QRect, pyqtSlot, QEasingCurve, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from Dynamic_display import Ui_Form
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5 import QtCore, QtGui, QtWidgets
class DynamicDisplayFunction(QWidget):
    """
    实现文章查询功能
    """

    def __init__(self):
        super(DynamicDisplayFunction, self).__init__()
        self.is_clear_start = False
        self.opacity2 = None
        self.is_right_btn = None
        self.opacity = None
        self.timer = None
        self.pictures = []
        self.pic_num = None
        self.pic_nums = None
        self.item = None
        self.scene = None
        self.anim = None
        self.sidebar_is_show = False
        self.Ui = Ui_Form()  # 实例化 Ui
        self.Ui.setupUi(self)  # 初始化Ui
        self.Ui.retranslateUi(self)
        self.setWindowTitle('动态展示')
        self.setWindowIcon(QIcon("logo.png"))
        self.setFixedSize(self.width(), self.height())
        self.Ui.Sidebar.move(0 - self.Ui.Sidebar.width(), self.Ui.Sidebar.y())  # 侧边栏左侧停靠
        self.Ui.Sidebar.raise_()
        self.Ui.btn_left.hide()
        self.Ui.btn_right.hide()
        '''界面居中显示'''
        # 获取屏幕坐标系
        screen = QDesktopWidget().screenGeometry()
        # 获取窗口坐标系
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft), int(newTop))

    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()
        if not self.sidebar_is_show and x <= 10 \
                and self.Ui.Sidebar.y() + self.Ui.Sidebar.height() >= y >= self.Ui.Sidebar.y():
            self.sidebar_show(True)
        elif x > self.Ui.Sidebar.x() + self.Ui.Sidebar.width() \
                or y < self.Ui.Sidebar.y() or self.Ui.Sidebar.y() + self.Ui.Sidebar.height() < y:
            self.sidebar_show(False)
        else:
            pass

    def sidebar_show(self, flag=False):
        if flag and not self.sidebar_is_show:
            self.sidebar_is_show = True
            self.anim = QPropertyAnimation(self.Ui.Sidebar, b"geometry")  # 定义动画
            self.anim.setDuration(800)  # 动画时间
            self.anim.setEasingCurve(QEasingCurve.InOutQuad)
            self.anim.setStartValue(
                QRect(self.Ui.Sidebar.x(), self.Ui.Sidebar.y(), self.Ui.Sidebar.width(), self.Ui.Sidebar.height()))
            self.anim.setEndValue(
                QRect(self.Ui.Sidebar.x() + self.Ui.Sidebar.width(), self.Ui.Sidebar.y(), self.Ui.Sidebar.width(),
                      self.Ui.Sidebar.height()))
            self.anim.start()
        elif not flag and self.sidebar_is_show:
            self.sidebar_is_show = False
            self.anim = QPropertyAnimation(self.Ui.Sidebar, b"geometry")
            self.anim.setDuration(800)
            self.anim.setEasingCurve(QEasingCurve.InOutQuad)
            self.anim.setStartValue(
                QRect(self.Ui.Sidebar.x(), self.Ui.Sidebar.y(), self.Ui.Sidebar.width(), self.Ui.Sidebar.height()))
            self.anim.setEndValue(
                QRect(0 - self.Ui.Sidebar.width(), self.Ui.Sidebar.y(), self.Ui.Sidebar.width(),
                      self.Ui.Sidebar.height()))
            self.anim.start()

    @pyqtSlot()
    def on_btn1_clicked(self):
        self.clear_start()
        self.clear_pic()
        self.Ui.btn_left.show()
        self.Ui.btn_right.show()
        self.pic_nums = 4
        self.pic_num = 0
        self.pictures = []
        for i in range(self.pic_nums):
            self.pictures.append(QLabel(self.Ui.picture_show))
            self.pictures[i].setAlignment(Qt.AlignCenter)
            self.pictures[i].setScaledContents(True)
        self.pictures[0].setPixmap(QPixmap("img/img_0.jpg"))
        self.pictures[0].move(int(self.Ui.picture_show.width()/2-200), int(self.Ui.picture_show.height()/2-125))
        self.pictures[1].setPixmap(QPixmap("img/img_1.jpg"))
        self.pictures[1].move(int(self.Ui.picture_show.width()/2-200), int(self.Ui.picture_show.height()/2-125))
        self.pictures[2].setPixmap(QPixmap("img/img_2.jpg"))
        self.pictures[2].move(int(self.Ui.picture_show.width()/2-200), int(self.Ui.picture_show.height()/2-125))
        self.pictures[3].setPixmap(QPixmap("img/img_3.jpg"))
        self.pictures[3].move(int(self.Ui.picture_show.width()/2-200), int(self.Ui.picture_show.height()/2-125))
        self.pic_show()

    @pyqtSlot()
    def on_btn2_clicked(self):
        self.clear_start()
        self.clear_pic()

        self.Ui.btn_left.show()
        self.Ui.btn_right.show()
        self.pic_nums = 3
        self.pic_num = 0
        self.pictures = []
        for i in range(self.pic_nums):
            self.pictures.append(QLabel(self.Ui.picture_show))
            self.pictures[i].setAlignment(Qt.AlignCenter)
            self.pictures[i].setScaledContents(True)
        # 图片地址（改）
        self.pictures[0].setPixmap(QPixmap("img/img_4.jpg"))
        self.pictures[0].move(int(self.Ui.picture_show.width() / 2 - 200), int(self.Ui.picture_show.height() / 2 - 125))
        self.pictures[1].setPixmap(QPixmap("img/img_5.jpg"))
        self.pictures[1].move(int(self.Ui.picture_show.width() / 2 - 200), int(self.Ui.picture_show.height() / 2 - 125))
        self.pictures[2].setPixmap(QPixmap("img/img_6.jpg"))
        self.pictures[2].move(int(self.Ui.picture_show.width() / 2 - 200), int(self.Ui.picture_show.height() / 2 - 125))
        self.pic_show()

    @pyqtSlot()
    def on_btn3_clicked(self):
        self.clear_start()
        self.clear_pic()
        self.Ui.btn_left.show()
        self.Ui.btn_right.show()
        self.pic_nums = 4
        self.pic_num = 0
        self.pictures = []
        for i in range(self.pic_nums):
            self.pictures.append(QLabel(self.Ui.picture_show))
            self.pictures[i].setAlignment(Qt.AlignCenter)
            self.pictures[i].setScaledContents(True)
        self.pictures[0].setPixmap(QPixmap("img/img_7.jpg"))
        self.pictures[0].move(int(self.Ui.picture_show.width() / 2 - 538), int(self.Ui.picture_show.height() / 2 - 250))
        self.pictures[1].setPixmap(QPixmap("img/img_8.jpg"))
        self.pictures[1].move(int(self.Ui.picture_show.width() / 2 - 538), int(self.Ui.picture_show.height() / 2 - 250))
        self.pictures[2].setPixmap(QPixmap("img/img_9.jpg"))
        self.pictures[2].move(int(self.Ui.picture_show.width() / 2 - 538), int(self.Ui.picture_show.height() / 2 - 250))
        self.pictures[3].setPixmap(QPixmap("img/img_10.jpg"))
        self.pictures[3].move(int(self.Ui.picture_show.width() / 2 - 538), int(self.Ui.picture_show.height() / 2 - 250))
        self.pic_show()

    @pyqtSlot()
    def on_btn4_clicked(self):
        self.clear_start()
        self.clear_pic()
        self.Ui.btn_left.show()
        self.Ui.btn_right.show()
        self.pic_nums = 3
        self.pic_num = 0
        self.pictures = []
        for i in range(self.pic_nums):
            self.pictures.append(QLabel(self.Ui.picture_show))
            self.pictures[i].setAlignment(Qt.AlignCenter)
            self.pictures[i].setScaledContents(True)
        self.pictures[0].setPixmap(QPixmap("img/img_11.jpg"))
        self.pictures[0].move(int(self.Ui.picture_show.width() / 2 - 538), int(self.Ui.picture_show.height() / 2 - 350))
        self.pictures[1].setPixmap(QPixmap("img/img_12.jpg"))
        self.pictures[1].move(int(self.Ui.picture_show.width() / 2 - 538), int(self.Ui.picture_show.height() / 2 - 350))
        self.pictures[2].setPixmap(QPixmap("img/img_13.jpg"))
        self.pictures[2].move(int(self.Ui.picture_show.width() / 2 - 538), int(self.Ui.picture_show.height() / 2 - 350))
        self.pic_show()

    @pyqtSlot()
    def on_btn_left_clicked(self):
        if self.pic_num == 0:
            self.pic_num = self.pic_nums - 1
        else:
            self.pic_num -= 1
        self.is_right_btn = False
        self.pic_show()

    @pyqtSlot()
    def on_btn_right_clicked(self):
        if self.pic_num == self.pic_nums - 1:
            self.pic_num = 0
        else:
            self.pic_num += 1
        self.is_right_btn = True
        self.pic_show()

    def pic_show(self):
        self.pictures[self.pic_num].raise_()
        self.pictures[self.pic_num].show()
        if self.is_right_btn:
            old_num = self.pic_num - 1
            old_num = self.pic_nums - 1 if old_num < 0 else old_num
        else:
            old_num = self.pic_num + 1
            old_num = 0 if old_num > self.pic_nums - 1 else old_num
        self.draw(self.pictures[self.pic_num], self.pictures[old_num])

    def draw(self, t, t2):
        self.Ui.btn_left.setEnabled(False)
        self.Ui.btn_right.setEnabled(False)
        # 设置透明度
        self.opacity = QGraphicsOpacityEffect()  # 透明度对象
        self.opacity.setOpacity(0)  # 初始化设置透明度为0，即完全透明
        t.setGraphicsEffect(self.opacity)  # 把标签的透明度设置为为self.opacity
        self.opacity.i = 1  # 用于记录透明度变化与循环次数

        # 设置旧图透明度
        self.opacity2 = QGraphicsOpacityEffect()  # 透明度对象
        self.opacity2.setOpacity(100)  # 初始化设置透明度为0，即完全透明
        t2.setGraphicsEffect(self.opacity2)  # 把标签的透明度设置为为self.opacity
        self.opacity2.i = 100  # 用于记录透明度变化与循环次数

        def timeout():  # 超时函数：改变透明度
            self.opacity.setOpacity(self.opacity.i / 100)
            self.opacity2.setOpacity(self.opacity2.i / 100)
            t.setGraphicsEffect(self.opacity)  # 改变标签透明度
            t2.setGraphicsEffect(self.opacity2)  # 改变标签透明度
            self.opacity.i += 1
            self.opacity2.i -= 1
            if self.opacity.i >= 100 or self.opacity2.i <= 0:  # 此时透明度为1，即不透明，控件已经完全显示出来了
                self.timer.stop()  # 计时器停止
                self.timer.deleteLater()
                self.Ui.btn_left.setEnabled(True)
                self.Ui.btn_right.setEnabled(True)
                # if self.is_right_btn:
                #     old_num = self.pic_num - 1
                #     old_num = 3 if old_num < 0 else old_num
                # else:
                #     old_num = self.pic_num + 1
                #     old_num = 0 if old_num > 3 else old_num
                # self.pictures[old_num].hide()
        self.timer = QTimer()  # 计时器
        self.timer.setInterval(10)  # 设置间隔时间，毫秒为单位
        self.timer.timeout.connect(timeout)  # 超时槽函数，每到达间隔时间，调用该函数
        self.timer.start()  # 计时器开始

    def clear_pic(self):
        for pic in self.pictures:
            pic.deleteLater()

    def clear_start(self):
        if not self.is_clear_start:
            self.Ui.label_3.deleteLater()
            self.Ui.label_4.deleteLater()
            self.is_clear_start = True

if __name__ == "__main__":
    import sys
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = DynamicDisplayFunction()
    MainWindow.showMaximized()
    sys.exit(app.exec_())