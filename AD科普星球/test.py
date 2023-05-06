from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import QPainter, QBrush, QImage, QColor, QPainterPath, QPen
from PyQt5.QtWidgets import *
import sys

Q = QApplication(sys.argv)


class myQWidget(QWidget):
    '''重写一个Qwidget'''

    def mouseMoveEvent(self, QMouseEvent):
        print('鼠标当前位置', QMouseEvent.pos())

    def paintEvent(self, QEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)  # 反锯齿
        '''渐变算法'''
        for i in range(0, 9):  # 边框重绘的次数，数值越大阴影越浓
            painter.setPen(QColor(0, 0, 0, 0))
            painter.setBrush(QColor(0, 0, 0, 6 + i))
            painter.drawRoundedRect(i, i, self.width() - 2 * i, self.height() - 2 * i, 0,
                                    0)  # 设置阴影的轮廓,4个与setGeometry参数一样，后面两个为窗口圆角
        # painter.setPen(QColor(0,0,0,0))
        painter.setBrush(QColor(255, 255, 255))
        painter.drawRoundedRect(10, 10, self.width() - 20, self.height() - 20, 0, 0)


class mylabel(QLabel):
    cur_Index = 0


class Test_Window(QWidget):
    def __init__(self):
        super(Test_Window, self).__init__()
        self.resize(800, 600)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.init_swiper()
        self.loop_init_label()

    def init_swiper(self):
        # self.l=QLabel(self)
        # self.l.setGeometry(200,100,400,400)
        # self.l.setStyleSheet("QLabel{background:red;}")
        self.left_btn = QPushButton(self)
        self.left_btn.setGeometry(0, 250, 40, 100)
        self.left_btn.clicked.connect(self.init_to_left_animation)
        self.right_btn = QPushButton(self)
        self.right_btn.setGeometry(self.width() - 40, 250, 40, 100)


    def loop_init_label(self):
        name = []
        for i in range(0, 3):
            label = 'label' + str(i)
            name.append(label)
        for i in range(0, len(name)):
            self.name = name
            self.name[i] = mylabel(self)
            self.name[i].cur_Index = i
            if i==0:
                self.name[i].setStyleSheet("QLabel{background:red;}")
                self.name[i].setGeometry(50, 150, 300,300)
                continue
            if i==1:
                self.name[i].setStyleSheet("QLabel{background:blue;}")
                self.name[i].setGeometry(200, 100, 400,400)

                continue
            if i==2:
                self.name[i].setStyleSheet("QLabel{background:yellow;}")
                self.name[i].setGeometry(450, 150, 300,300)
                continue

            print(self.name[i],self.name[i].cur_Index)
        self.name_=name
        self.name_[1].raise_()
        print('标签初始化完毕')

    def init_to_left_animation(self):
        """向左移动动画"""


        for i in range(0, len(self.name)):
            print(self.name[i].cur_Index)
            if self.name[i].cur_Index == 1:
                print('将索引为1的标签置中')
                self.name[i].setGeometry(200, 100, 400, 400)
                self.anim1=QPropertyAnimation(self.name[i],b'geometry')
                self.anim1.setDuration(300)
                self.anim1.setStartValue(QRect(450, 150, 300, 300))
                self.anim1.setEndValue(QRect(200, 100, 400, 400))
                self.anim1.start()
                self.name[i].cur_Index = 0
                self.name[i].raise_()
                continue

            if self.name[i].cur_Index == 0:
                print('将索引为0的标签置左')
                self.anim0=QPropertyAnimation(self.name[i],b'geometry')
                self.anim0.setDuration(300)
                self.anim0.setStartValue(QRect(200, 100, 400, 400))
                self.anim0.setEndValue(QRect(50, 150, 300, 300))
                self.anim0.start()
                self.name[i].cur_Index = 2
                continue

            if self.name[i].cur_Index == 2:
                print('将索引为2的标签置右')
                self.anim2=QPropertyAnimation(self.name[i],b'geometry')
                self.anim2.setDuration(300)
                self.anim2.setStartValue(QRect(50, 150, 300, 300))
                self.anim2.setEndValue(QRect(450, 150, 300, 300))
                self.anim2.start()
                self.name[i].cur_Index = 1
                continue


        print('动画结束')


if __name__ == '__main__':
    app = Q
    win = Test_Window()
    win.show()
    sys.exit(app.exec_())
