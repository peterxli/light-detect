import shutil
import os
import random

'''
从"img_path"文件夹随机选取30%的图片移动到"copy_to_path"文件夹中
'''

ROOT_DIR = os.path.abspath("../")
img_path = os.path.join(ROOT_DIR, "/home/hualai/ponder/lxy/light_detect/detect_object/dataset/20200820dataset/test64/dark")
copy_to_path = os.path.join(ROOT_DIR, "/home/hualai/ponder/lxy/light_detect/detect_object/dataset/20200820dataset/validation64/dark")
imglist = os.listdir(img_path)
random_imglist = random.sample(imglist, int(1/3 * len(imglist)))  # 随机选取30%
for img in random_imglist:
    # 图片复制到另一个文件夹
    shutil.copy(os.path.join(img_path, img), os.path.join(copy_to_path, img))
    os.remove(os.path.join(img_path, img))  # 并删除原有文件