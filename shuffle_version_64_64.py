# -*- coding: utf-8 -*-

import os
from shutil import copyfile
import random
import shutil
import cv2




def process():
    dir_path = '/Users/peterxli/Desktop/20200628'
    count = 0
    
    sub_dirs = os.listdir(dir_path + '/all_data')
    
    os.mkdir(dir_path + '/train_data')
    os.mkdir(dir_path + '/test_data')
    
    for sub_dir in sub_dirs:
        sub_dir_path = dir_path + '//all_data//' + sub_dir
        
        os.mkdir(dir_path + '/train_data/' + sub_dir)
        os.mkdir(dir_path + '/test_data/' + sub_dir)
        
        files = os.listdir(sub_dir_path)
        for filename in files:
            count = count + 1
            
            srcFilePath = sub_dir_path + '//' + filename
            
            trainFilePath = dir_path + '//train_data//' + sub_dir + '//' + filename
            testFilePath = dir_path + '//test_data//' + sub_dir + '//' + filename
            
            src_img = cv2.imread(srcFilePath)
            resize_img = cv2.resize(src_img, (64, 64), cv2.INTER_AREA)
            cv2.imwrite(dir_path,resize_img)
            if count % 5 == 0:
                cv2.imwrite(testFilePath, resize_img)
            else:
                cv2.imwrite(trainFilePath, resize_img)     

def change_process():
    image_size = 64                          
    source_path="/home/hualai/ponder/lxy_project/light_detect/detect_object/dataset/0922crop/"
    target_path="/home/hualai/ponder/lxy_project/light_detect/detect_object/dataset/20200922dataset/train64/light/"
    # i = 0
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    image_list = os.listdir(source_path)      
    for filename in image_list:
        # i=i+1
        image_source = cv2.imread(source_path+filename)
        print(source_path + filename)
        image = cv2.resize(image_source, (image_size, image_size), cv2.INTER_AREA)
        cv2.imwrite(target_path + filename ,image)
    print("批量处理完成")

# def moveFile(fileDir,tarDir):
#     # fileDir = '/home/hualai/ponder/lxy/light_detect/detect_object/dataset/20200820dataset/train64/dark'    # 源图片文件夹路径
#     # tarDir = '/home/hualai/ponder/lxy/light_detect/detect_object/dataset/20200820dataset/test64/dark/'     # 移动到新的文件夹路径
#     pathDir = os.listdir(fileDir)    # 取图片的原始路径
#     sample = random.sample(pathDir, 45)  # 随机选取picknumber数量的样本图片, picknumber=200
#     print (sample)
#     for name in sample:
#         shutil.move(fileDir+name, tarDir+name)


def change_size():
    rootdir = r'D:\\数据集\\20200713'
    for parent, dirnames, filenames in os.walk(rootdir):#遍历每一张图片
        for filename in filenames:
            print('parent is :' + parent)
            print('filename is :' + filename)
            currentPath = os.path.join(parent, filename)
            print('the fulll name of the file is :' + currentPath)
            img = Image.open(currentPath)
            print (img.format, img.size, img.mode)
            #img.show()
            box1 = (179, 276, 889, 920)#设置左、上、右、下的像素
            image1 = img.crop(box1) # 图像裁剪
            change_process(image1)
            image1.save(r"D:\\数据集\\off"+'\\'+filename) #存储裁剪得到的图像




def random_moveFile(fileDir):
    fileDir = '/home/hualai/ponder/lxy/light_detect/detect_object/dataset/20200820dataset/train64/dark'
    tarDir = '/home/hualai/ponder/lxy/light_detect/detect_object/dataset/20200820dataset/test64/dark/'
    pathDir = os.listdir(fileDir)
    sample = random.sample(pathDir,45)
    print(sample)
    for name in sample:
        shutil.move(fileDir+name,tarDir+name)
    return


def main():   
    # process()
    change_process()
    # moveFile(fileDir)
    # random_moveFile()
    

if __name__ == '__main__':
    main()
