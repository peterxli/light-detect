import os
import random
import shutil


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.bmp')]


def getData(src_path):
    dest_dir = './test'  # 这个文件夹需要提前建好
    img_list = get_imlist(src_path)
    random.shuffle(img_list)
    le = int(len(img_list) * 0.8)  # 这个可以修改划分比例
    for f in img_list[le:]:
        shutil.move(f, dest_dir)


getData('./test_data2')