# -*- coding: utf-8 -*-

import os
import time
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.pyplot as plt

from torch.autograd import Variable
import numpy as np
import torch.nn as nn

import cv2
import dlib
from PIL import Image
import requests
import json

import logging
import shutil
import glob

##################################################################
''' Logger模块 '''
##################################################################

logger = logging.getLogger("logger")
logger.setLevel(level = logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if not logger.handlers:
    fileHandler = logging.FileHandler("light.log")
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    

##################################################################
''' 准备数据集 '''
##################################################################

# 数据增强
transform = transforms.Compose(
    [transforms.RandomCrop((56, 56)),
     transforms.Resize((64, 64)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(0.1),
     transforms.RandomRotation(45),
     transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
     transforms.RandomGrayscale(0.1),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

# 训练集
train_dataset = torchvision.datasets.ImageFolder(root = '/home/hualai/ponder/lxy_project/light_detect/detect_object/dataset/20200922dataset/train64',
                                                 transform = transform)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = 8,
                                           shuffle = True,
                                           num_workers = 0)

logger.info('train_set: %s' % str(train_dataset.class_to_idx))

# 验证集
valid_dataset = torchvision.datasets.ImageFolder(root = '/home/hualai/ponder/lxy_project/light_detect/detect_object/dataset/20200922dataset/test64',
                                                 transform = transform)

valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset,
                                           batch_size = 8,
                                           shuffle = True,
                                           num_workers = 0)

logger.info('valid_set: %s' % str(valid_dataset.class_to_idx))

# 测试集
test_dataset = torchvision.datasets.ImageFolder(root = '/home/hualai/ponder/lxy_project/light_detect/detect_object/dataset/20200922dataset/validation64',
                                                transform = transform)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = 8,
                                          shuffle = False,
                                          num_workers = 0)

logger.info('test_set: %s' % str(test_dataset.class_to_idx))


####################################################################
''' 图像分类 '''
####################################################################

classes = ('dark', 'bright')


####################################################################
''' 构建卷积神经网络MobileNet '''
####################################################################

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
    cfg = [64, (128,2), 128, (256,2), 256,
           (512,2), 512, 512, 512, 512,
           512, (1024,2), 1024]
    
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
        self.linear = nn.Linear(64*64, num_classes)

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


#####################################################################
''' 卷积神经网络的训练 '''
#####################################################################

net = MobileNet()
if torch.cuda.is_available():
    net.cuda()

def train():
    learning_rate = 0.0001
    momentum = 0.9
    num_epoches = 250
    
    # 定义优化方法，随机梯度下降
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # 交叉熵损失函数(适用于多分类任务)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epoches):
        logger.info('----- Epoch %d train starts now -----' % (epoch + 1))
        #images 训练集的图像 labels 训练集的标签
        train_loss = 0.0
        for batch_index, (images, labels) in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
            # 将数据输入到网络，得到第一轮网络前向传播的预测结果
            output = net(images)
            # 预测结果通过交叉熵计算损失
            loss = criterion(output, labels)
            # 将梯度变为0
            optimizer.zero_grad()
            # 误差反向传播
            loss.backward()
            # 随机梯度下降方法优化权重
            optimizer.step()
            # 查看网络训练状态以及收敛情况
            train_loss += loss.item()
            if batch_index % 10 == 0:
                print('[%3d, %4d], loss = %.5f' % (epoch + 1, batch_index + 1, train_loss / 10))
                train_loss = 0.0
        # 计算训练集预测效果
        print('---------- Train dataset predicts ----------')
        networkTotalAccuracy(train_loader)
        networkClassAccuracy(train_loader)
        
        # 计算验证集预测效果
        print('---------- Valid dataset predicts ----------')
        networkTotalAccuracy(valid_loader)
        networkClassAccuracy(valid_loader)
        
        # 计算测试集预测效果
        print('---------- Test dataset predicts ----------')
        networkTotalAccuracy(test_loader)
        networkClassAccuracy(test_loader)
        
        if not os.path.isdir('/home/hualai/ponder/lxy/light_detect/detect_object/checkpoint092201'):
            os.mkdir('/home/hualai/ponder/lxy/light_detect/detect_object/checkpoint092201')
        torch.save(net, '/home/hualai/ponder/lxy/light_detect/detect_object/checkpoint092201/light_classifier_64_64_%d.pt' % (epoch+1))
        
        logger.info('Saving epoch %d model successfully!' % (epoch + 1))


#####################################################################
''' 批量计算整个数据集预测效果 '''
#####################################################################

def networkTotalAccuracy(dataset_loader):    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset_loader:
            images, labels = data
            if torch.cuda.is_available():
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
            with torch.no_grad():
                outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    logger.info('Accuracy of the network: %d %% (%d/%d)' % 
          (100 * correct / total, correct, total))


#####################################################################
''' 计算每个类的预测效果 '''
#####################################################################

def networkClassAccuracy(dataset_loader):
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in dataset_loader:
            images, labels = data
            if torch.cuda.is_available():
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
            with torch.no_grad():
                outputs = net(images)
            __, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    for i in range(len(classes)):
        logger.info('Accuracy of %7s: %2d %%' % 
              (classes[i], 100*class_correct[i]/class_total[i]))


#####################################################################
''' 处理视频 '''
#####################################################################
def send(result):
    try:
        r = requests.post(url,data = json.dumps(result),timeout = 1)
        r.raise_for_status()
        r.encoding=r.apparent_encoding
        return r.text
        print(r.text)
        print(r.status_code)
    except:
        return "产生异常"

def lightsAnalysis(classes_list):
    light_stats = [0, 0, 0, 0]
    maxIndex = 0
    maxVal = 0

    for class_id in classes_list:
        light_stats[class_id] = light_stats[class_id] + 1

    for i in range(len(light_stats)):
        if light_stats[i] > maxVal:
            maxVal = light_stats[i]
            maxIndex = i

    score = maxVal / float(len(classes_list))
    return {'class_id': maxIndex, 'class_name': classes[maxIndex], 'score': score}


def processVideo(flag,image):
    detector_model = dlib.simple_object_detector('/home/hualai/ponder/lxy/light_detect/detect_object/detector.svm')
    classifier_model = torch.load('/home/hualai/ponder/lxy/light_detect/detect_object/checkpoint/light_classifier_64_64_196.pt')
    light_box = {'left': 0.0, 'top': 0.0, 'right': 0.0, 'bottom': 0.0}

    classes_list = []

    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('/home/hualai/ponder/lxy/light_detect/detect_object/testvideo/test01.mov')

    # Off status by default
    lastLightsClassID = 0

    time_start = time.time()

    # while True:
        # ret, frame = cap.read()
    ret,frame = flag,image
    # if not ret:
    #     print('Video open failed')
    #     break

    debug_frame = frame
    label_list = []
    light_rects = detector_model(debug_frame, 0)
    for index, light in enumerate(light_rects):
        light_box['left'] = light.left()
        light_box['top'] = light.top()
        light_box['right'] = light.right()
        light_box['bottom'] = light.bottom()
        cv2.rectangle(debug_frame,
                      (light_box['left'], light_box['top']),
                      (light_box['right'], light_box['bottom']),
                      (0, 255, 0),
                      3)
        crop_frame = debug_frame[light_box['top']:light_box['bottom'],
                                 light_box['left']:light_box['right']]
        if crop_frame.shape[0]==0 or  crop_frame.shape[1]==0:
            continue
        resize_frame = cv2.resize(crop_frame, (64, 64), cv2.INTER_AREA)

        img = Image.fromarray(resize_frame)
        img_tensor = transform(img)

        if torch.cuda.is_available():
            img_tensor = img_tensor.view(1, 3, 64, 64).cuda()
        else:
            img_tensor = img_tensor.view(1, 3, 64, 64)

        with torch.no_grad():
            classifier_model.eval()
            output = classifier_model(img_tensor)

            _, indices = torch.sort(output, descending=True)
            percentage = torch.nn.functional.softmax(output, dim=1)[0]
            # print([(classes[i], percentage[i].item()) for i in indices[0][:2]])

        max_score_index = indices[0][:1][0].item()
        if percentage[max_score_index] >= 0.7:
            text = 'category: %s %.5f' % (classes[indices[0][:1][0]], percentage[indices[0][:1][0]].item())
            cv2.putText(debug_frame, text, (light_box['right'], light_box['bottom']), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            # print(classes[indices[0][:1][0]])
            label_list.append(classes[indices[0][:1][0]])
            # print(label_list)
    classes_list.append(label_list)

    print(classes_list)
    if len(classes_list) > 20:
        out_list = classes_list.pop(0)
        if 'dark' in out_list:
            text = "photo is wrong !"
        else:
            text = 'good light !'
        cv2.putText(debug_frame, text, (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        return debug_frame
    return image
    #     result = lightsAnalysis(classes_list)
    #     nowLightsClassID = result['class_id']
    #     nowLightsScore = result['score']
    #     print("----- Predict lights: %s" % nowLightsClassID)
    #     print("----- Predict scores: %s" % nowLightsScore)
#     cv2.imshow('Wheel Demo', debug_frame)
#     cv2.waitKey(1)
# #
# cap.release()
# cv2.destroyAllWindows()

#peterxli

def cut_BoardPhoto():
    detector_model = dlib.simple_object_detector('detector.svm')
    board_box = {'left': 0.0, 'top': 0.0, 'right': 0.0, 'bottom': 0.0}
    current_path = os.getcwd()
    test_folder = current_path + '/20200820/'
    for imgName in glob.glob(test_folder + '*.jpg'):
        name = os.path.basename(imgName)
        print("Processing file: {}".format(imgName))
        img = cv2.imread(imgName, cv2.IMREAD_COLOR)
        b, g, r = cv2.split(img)
        img2 = cv2.merge([r, g, b])
        board_rects = detector_model(img2)
        count = 0
        for index, board in enumerate(board_rects):
            count = count + 1
            print(
                'board {}; left {}; top {}; right {}; bottom {}'.format(index, board.left(), board.top(), board.right(),
                                                                        board.bottom()))
            board_box['left'] = board.left()
            board_box['top'] = board.top()
            board_box['right'] = board.right()
            board_box['bottom'] = board.bottom()
            cv2.rectangle(img, (board_box['left'], board_box['top']), (board_box['right'], board_box['bottom']), 3)
            crop_frame = img[board_box['top']:board_box['bottom'],
                            board_box['left']:board_box['right']]
            print(count)
            # cv2.imshow('name',crop_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            print(cv2.imwrite('/Users/peterxli/Hualai/detect_object/crop_frame/' +str(count) + name,crop_frame))


def detectPhoto():
    cut_BoardPhoto()
    detector_model = dlib.simple_object_detector('detector.svm')
    classifier_model = torch.load()
    light_box = {'left': 0.0, 'top': 0.0, 'right': 0.0, 'bottom': 0.0}
    light_box_flag = False
    photo_Path = '/Users/peterxli/Hualai/detect_object/crop_frame'
    photoDir = os.listdir(photo_Path)
    for imgName in glob.glob(photo_Path + '*.jpg'):
        print("Processing file: {}".format(imgName))
        img = cv2.imread(imgName, cv2.IMREAD_COLOR)
        b, g, r = cv2.split(img)
        img2 = cv2.merge([r, g, b])
        if not light_box_flag:
            light_rects = detector_model(img2)
            for index, light in enumerate(light_rects):
                light_box['left'] = light.left()
                light_box['top'] = light.top()
                light_box['right'] = light.right()
                light_box['bottom'] = light.bottom()
                light_box_flag = True
        if light_box_flag:
            cv2.rectangle(img,
                          (light_box['left'], light_box['top']),
                          (light_box['right'], light_box['bottom']),
                          (0, 255, 0),
                          3)
            crop_frame = img[light_box['top']:light_box['bottom'],
                         light_box['left']:light_box['right']]
            resize_frame = cv2.resize(crop_frame, (64, 64), cv2.INTER_AREA)
            img = Image.fromarray(resize_frame)
            img_tensor = transforms.ToTensor(img)
            if torch.cuda.is_available():
                img_tensor = img_tensor.view(1, 3, 64, 64).cuda()
            else:
                img_tensor = img_tensor.view(1, 3, 64, 64)
            with torch.no_grad():
                classifier_model.eval()
                output = classifier_model(img_tensor)

                _, indices = torch.sort(output, descending=True)
                percentage = torch.nn.functional.softmax(output, dim=1)[0]
                # print([(classes[i], percentage[i].item()) for i in indices[0][:2]])
            max_score_index = indices[0][:3][0].item()
            if percentage[max_score_index] >= 0.7:
                text = 'category: %s %.5f' % (classes[indices[0][:3][0]], percentage[indices[0][:3][0]].item())
                cv2.putText(img, text, (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        cv2.imshow('Light Demo', img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()



def detect_Unmatched_img():
    fileDir = "D:\\Pythoncode\\wheel_detect\\wheel_detect\\20200714\\test_data\\off\\"
    tarDir = 'D:\\Pythoncode\\wheel_detect\\wheel_detect\\20200714\\test_data\\reoff\\'
    classname = 'off'
    # 加载分类器模型
    classifier_model = torch.load('D:\\Pythoncode\\wheel_detect\\wheel_detect\\20200714\\checkpoint\\wheel_classifier_64_64_200.pt',map_location='cpu')

    imageList = os.listdir(fileDir)
    # print(imageList)
    for imgName in imageList:
        imgPath = os.path.join(fileDir,imgName)
        # print(imgPath)
        imgPhoto = cv2.imread(imgPath,cv2.IMREAD_COLOR)
        img = Image.fromarray(imgPhoto)
        img_tensor = transform(img)
        if torch.cuda.is_available():
            img_tensor = img_tensor.view(1, 3, 64, 64).cuda()
        else:
            img_tensor = img_tensor.view(1, 3, 64, 64)
        with torch.no_grad():
            classifier_model.eval()
            output = classifier_model(img_tensor)
                
            _, indices = torch.sort(output, descending=True)
            percentage = torch.nn.functional.softmax(output, dim=1)[0]

        max_score_index = indices[0][:3][0].item()
        if percentage[max_score_index] >= 0.7:
            text = classes[indices[0][:3][0]]
            print(text)
        if text != classname:
            shutil.copy(fileDir+imgName, tarDir+text+imgName)


#####################################################################
''' 预测和评估'''
#####################################################################

def main():
    # train()
    processVideo()
    # detect_Unmatched_img()
    # detectPhoto()
    # cut_BoardPhoto()
if __name__ == '__main__':
    main()
    