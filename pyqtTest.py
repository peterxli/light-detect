from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import mobilenet_light_version_64_64
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                               padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2,
    # by default conv stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256,
           (512, 2), 512, 512, 512, 512,
           512, (1024, 2), 1024]

    # 定义卷积神经网络需要的元素
    def __init__(self, num_classes=2):
        super(MobileNet, self).__init__()
        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # 拟规范化层加速神经网络收敛过程
        self.bn1 = nn.BatchNorm2d(32)
        # 定义卷积层
        self.layers = self._make_layers(in_channels=32)
        # 最后一个全连接层用作分类
        self.linear = nn.Linear(64 * 64, num_classes)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 第一个卷积层首先经过RELU做激活
        out = F.relu(self.bn1(self.conv1(x)))
        # 定义中间卷积层
        out = self.layers(out)
        # 池化层
        out = F.avg_pool2d(out, 2)
        # 堆特征层Tensor维度进行变换
        out = out.view(out.size(0), -1)
        # 卷积神经网络的特征经过最后一次全连接层操作，得到最终分类结果
        out = self.linear(out)
        return out



class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)  # 父类的构造函数

        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap0 = cv2.VideoCapture()  # 视频流
        self.cap1 = cv2.VideoCapture()  # 视频流
        self.CAM_NUM0 = 1  # 为0时表示视频流来自笔记本内置摄像头
        self.CAM_NUM1 = 0  # 为0时表示视频流来自笔记本内置摄像头
        self.set_ui()  # 初始化程序界面
        self.slot_init()  # 初始化槽函数

    '''程序界面布局'''

    def set_ui(self):
        self.__layout_main = QtWidgets.QVBoxLayout()  # 总布局

        # self.__layout_data_show = QtWidgets.QHBoxLayout()  # 数据(视频)显示布局

        '''信息显示'''
        self.__layout_show_camera = QtWidgets.QHBoxLayout()
        self.label_show_camera0 = QtWidgets.QLabel()  # 定义显示视频的Label
        self.label_show_camera0.setFixedSize(400, 300)  # 给显示视频的Label设置大小为641x481
        self.label_show_camera1 = QtWidgets.QLabel()  # 定义显示视频的Label
        self.label_show_camera1.setFixedSize(400, 300)  # 给显示视频的Label设置大小为641x481
        self.__layout_show_camera.addWidget(self.label_show_camera0)
        self.__layout_show_camera.addWidget(self.label_show_camera1)
        # 按键布局
        self.__layout_fun_button = QtWidgets.QHBoxLayout()
        '''把按键加入到按键布局中'''
        self.button_open_camera = QtWidgets.QPushButton('打开相机')  # 建立用于打开摄像头的按键
        self.button_close = QtWidgets.QPushButton('退出')  # 建立用于退出程序的按键
        self.button_open_camera.setMinimumHeight(50)  # 设置按键大小
        self.button_close.setMinimumHeight(50)
        self.button_close.move(10, 100)  # 移动按键
        self.__layout_fun_button.addWidget(self.button_open_camera)  # 把打开摄像头的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_close)  # 把退出程序的按键放到按键布局中


        '''把某些控件加入到总布局中'''
        self.__layout_main.addLayout(self.__layout_show_camera)  # 把用于显示视频的Label加入到总布局中
        self.__layout_main.addLayout(self.__layout_fun_button)  # 把按键布局加入到总布局中
        '''总布局布置好后就可以把总布局作为参数传入下面函数'''
        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件

    '''初始化所有槽函数'''

    def slot_init(self):
        self.button_open_camera.clicked.connect(
            self.button_open_camera_clicked)  # 若该按键被点击，则调用button_open_camera_clicked()
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()
        self.button_close.clicked.connect(self.close)  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序

    '''槽函数之一'''

    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            self.flag0 = self.cap0.open(self.CAM_NUM0)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if self.flag0 == False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机0于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('关闭相机')
            self.flag1 = self.cap1.open(self.CAM_NUM1)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if self.flag1 == False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机1于电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('关闭相机')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap0.release()  # 释放视频流
            self.label_show_camera0.clear()  # 清空视频显示区域
            self.cap1.release()  # 释放视频流
            self.label_show_camera1.clear()  # 清空视频显示区域
            self.button_open_camera.setText('打开相机')

    def show_camera(self):
        if self.flag0:
            flag, self.image0 = self.cap0.read()  # 从视频流中读取
            show0 = mobilenet_light_version_64_64.processVideo(flag,self.image0)
            show0 = cv2.resize(show0, (400, 300))  # 把读到的帧的大小重新设置为 640x480
            show0 = cv2.cvtColor(show0, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
            showImage0 = QtGui.QImage(show0.data, show0.shape[1], show0.shape[0],
                                     QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.label_show_camera0.setPixmap(QtGui.QPixmap.fromImage(showImage0))  # 往显示视频的Label里 显示QImage
        if self.flag1:
            flag, self.image1 = self.cap1.read()  # 从视频流中读取
            show1 = mobilenet_light_version_64_64.processVideo(flag,self.image1)
            if show1.shape[0] == 0 or show1.shape[1] == 0:
                return
            show1 = cv2.resize(show1, (400, 300))  # 把读到的帧的大小重新设置为 640x480
            show1 = cv2.cvtColor(show1, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
            showImage1 = QtGui.QImage(show1.data, show1.shape[1], show1.shape[0],
                                     QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.label_show_camera1.setPixmap(QtGui.QPixmap.fromImage(showImage1))  # 往显示视频的Label里 显示QImage

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    ui = Ui_MainWindow()  # 实例化Ui_MainWindow
    ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())    