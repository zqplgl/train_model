import os
import shutil
import cv2
import numpy as np

def rename_data():
    pic_path = [line.strip() for line in open("./pic_path.txt").readlines()]
    pic_save_dir = "/media/zqp/data/data/patentimage/"

    pic_num = len(pic_path)
    index = 0
    for path in pic_path[:2000]:
        shutil.copy(path, pic_save_dir + "%08d.jpg"%index)
        index+=1

        if index%100==0:
            print("process*********%s/%s"%(index, pic_num))

def resize_data():
    pic_path = [line.strip() for line in open("./pic_path.txt").readlines()]
    pic_num = len(pic_path)
    index = 0

    for path in pic_path:
        im = cv2.imread(path,0)
        h,w = im.shape

        max_side = max(w,h)
        im_resize = (np.ones((max_side,max_side))*255).astype(np.uint8)
        if h<w:
            up = int((w - h)/2+0.7)
            im_resize[up:up+h] = im

        elif w<h:
            left = int((h - w)/2+0.7)
            im_resize[:,left:left+w] = im

        cv2.imwrite(path,im_resize)
        index += 1

        if index%100==0:
            print("processed ******************%s/%s"%(index, pic_num))

if __name__=="__main__":
    resize_data()

