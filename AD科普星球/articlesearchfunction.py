from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget
from article_search import Ui_Form
import pandas as pd
import webbrowser
from fuzzywuzzy import fuzz
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon, QFont, QPixmap
import sys
class ArticleSearchFunction(QWidget):
    """
    实现文章查询功能
    """

    def __init__(self):
        super(ArticleSearchFunction, self).__init__()
        self.data = None
        self.cur_page = 1
        self.pages = 0
        self.setWindowIcon(QIcon("logo.png"))
        self.Ui = Ui_Form()  # 实例化 Ui
        self.Ui.setupUi(self)  # 初始化Ui
        self.Ui.retranslateUi(self)
        self.setWindowTitle('科普文章查询')
        #self.setFixedSize(self.width(), self.height())
        self.data = pd.read_excel('data/article.xlsx', sheet_name=0)
        self.data['match_rate'] = 0
        self.pages = int(self.data.shape[0] / 7)
        self.show_data()

    def show_data(self):
        self.Ui.page_current.setText(str(self.cur_page))
        self.Ui.url1.setText(self.data.at[(self.cur_page - 1) * 7 + 0, "title"])
        self.Ui.url2.setText(self.data.at[(self.cur_page - 1) * 7 + 1, "title"])
        self.Ui.url3.setText(self.data.at[(self.cur_page - 1) * 7 + 2, "title"])
        self.Ui.url4.setText(self.data.at[(self.cur_page - 1) * 7 + 3, "title"])
        self.Ui.url5.setText(self.data.at[(self.cur_page - 1) * 7 + 4, "title"])
        self.Ui.url6.setText(self.data.at[(self.cur_page - 1) * 7 + 5, "title"])
        self.Ui.url7.setText(self.data.at[(self.cur_page - 1) * 7 + 6, "title"])
        self.Ui.source1.setText(self.data.at[(self.cur_page - 1) * 7 + 0, "source"])
        self.Ui.source2.setText(self.data.at[(self.cur_page - 1) * 7 + 1, "source"])
        self.Ui.source3.setText(self.data.at[(self.cur_page - 1) * 7 + 2, "source"])
        self.Ui.source4.setText(self.data.at[(self.cur_page - 1) * 7 + 3, "source"])
        self.Ui.source5.setText(self.data.at[(self.cur_page - 1) * 7 + 4, "source"])
        self.Ui.source6.setText(self.data.at[(self.cur_page - 1) * 7 + 5, "source"])
        self.Ui.source7.setText(self.data.at[(self.cur_page - 1) * 7 + 6, "source"])

    @pyqtSlot()
    def on_search_clicked(self):
        title_to_find = self.Ui.lineEdit.text()
        for index, row in self.data.iterrows():
            self.data.loc[index, 'match_rate'] = fuzz.partial_ratio(title_to_find, row['title'])
        self.data.sort_values(by="match_rate", inplace=True, ascending=False)
        self.data = self.data.reset_index(drop=True)
        self.on_page_start_clicked()

    @pyqtSlot()
    def on_url1_clicked(self):
        webbrowser.open_new_tab(self.data.at[(self.cur_page - 1) * 7 + 0, "url"])

    @pyqtSlot()
    def on_url2_clicked(self):
        webbrowser.open_new_tab(self.data.at[(self.cur_page - 1) * 7 + 1, "url"])

    @pyqtSlot()
    def on_url3_clicked(self):
        webbrowser.open_new_tab(self.data.at[(self.cur_page - 1) * 7 + 2, "url"])

    @pyqtSlot()
    def on_url4_clicked(self):
        webbrowser.open_new_tab(self.data.at[(self.cur_page - 1) * 7 + 3, "url"])

    @pyqtSlot()
    def on_url5_clicked(self):
        webbrowser.open_new_tab(self.data.at[(self.cur_page - 1) * 7 + 4, "url"])

    @pyqtSlot()
    def on_url6_clicked(self):
        webbrowser.open_new_tab(self.data.at[(self.cur_page - 1) * 7 + 5, "url"])

    @pyqtSlot()
    def on_url7_clicked(self):
        webbrowser.open_new_tab(self.data.at[(self.cur_page - 1) * 7 + 6, "url"])

    # 下一页
    @pyqtSlot()
    def on_page_behind_clicked(self):
        if self.pages is None:
            return
        if self.cur_page == self.pages:
            return
        self.cur_page += 1
        self.show_data()

    # 前一页
    @pyqtSlot()
    def on_page_front_clicked(self):
        if self.pages is None:
            return
        if self.cur_page == 1:
            return
        self.cur_page -= 1
        self.show_data()

    # 首页
    @pyqtSlot()
    def on_page_start_clicked(self):
        if self.pages is None:
            return
        self.cur_page = 1
        self.show_data()

    # 尾页
    @pyqtSlot()
    def on_page_end_clicked(self):
        if self.pages is None:
            return
        self.cur_page = self.pages
        self.show_data()

    # 对应页直接跳转
    @pyqtSlot()
    def on_page_confirm_clicked(self):
        page_in = self.Ui.page_input.text()
        if len(page_in) == 0 or (not page_in.isdigit()) or self.pages is None:
            return
        p = int(page_in)
        if p <= 1:
            self.cur_page = 1
        elif p >= self.pages:
            self.cur_page = self.pages
        else:
            self.cur_page = p
        self.show_data()
