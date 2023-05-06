from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog
import os
import shutil

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.button = QPushButton("Upload file", self)
        self.button.clicked.connect(self.show_file_dialog)

    def show_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a file to upload", ".", "All Files (*)")
        if file_path:
            # 指定保存文件的目录
            save_dir = 'user_feature'

            # 从文件路径中获取文件名
            file_name = os.path.basename(file_path)

            # 拼接保存文件的完整路径
            save_path = os.path.join(save_dir, file_name)

            # 复制文件到指定目录中
            shutil.copyfile(file_path, save_path)

if __name__ == '__main__':
    app = QApplication([])
    widget = MyWidget()
    widget.show()
    app.exec_()
