import os
import random
import shutil

def run():
    pic_path = [line.strip() for line in open("./pic_path.txt").readlines()]
    pic_save_root_dir = "/media/zqp/data/train_data/patent_retrieval/"

    class_index = 0
    random.shuffle(pic_path)

    for path in pic_path:
        pic_name = os.path.basename(path)
        pic_save_dir = pic_save_root_dir+"%06d/"%(class_index)

        os.makedirs(pic_save_root_dir+"%06d"%(class_index))
        shutil.copy(path, pic_save_dir+"%06d.jpg"%class_index)

        class_index += 1
        if(class_index>=1000):
            break;

def run_1():
    pic_save_root_dir = "/media/zqp/data/train_data/patent_retrieval/"
    folder_names = os.listdir(pic_save_root_dir)
    folder_names = sorted(folder_names)

    for folder_name in folder_names:
        pic_dir = pic_save_root_dir+folder_name

        print(folder_name,len(os.listdir(pic_dir)))

        #if len(os.listdir(pic_dir))<1:
            #print(pic_dir)

def run_2():
    pic_path = [line.strip() for line in open("./pic_path.txt").readlines()]
    pic_save_root_dir = "/media/zqp/data/train_data/patent_retrieval/"
    folder_names = os.listdir(pic_save_root_dir)


if __name__=="__main__":
    #run()
    run_1()

