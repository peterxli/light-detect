import numpy as np
import cv2
import os

def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + (1-alpha) * blank + beta
    dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)
    return dst
def open_camera():
    cap = cv2.VideoCapture(0)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()  ##ret返回布尔量
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame1 = Contrast_and_Brightness(1, -1, frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('/Users/peterxli/Hualai/detect_object/20200819/' + str(i) + '.jpg', frame1)
            i += 1
        cv2.imshow('frame1', frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def change_name():

    input_dir = "/Users/peterxli/Hualai/detect_object/20200819/"
    output_dir = "/Users/peterxli/Hualai/detect_object/20200819/"

    if os.path.exists(output_dir):
        os.mkdir(output_dir)

    for fname in os.listdir(input_dir):
        if "_0" in fname:
            print(str(fname))
        else:
            fname0 = str(fname).split(".")[0]
            fname_new = fname0 + "_1" + ".tif"
            print("修改后：", fname_new)
            # os.renames(文件路径/旧文件名，文件路径/新文件名)
            os.renames("/Users/peterxli/Hualai/detect_object/20200819" + fname, "/Users/peterxli/Hualai/detect_object/20200819" + fname_new)

def main():
    change_name()


if __name__ == '__main__':
    main()