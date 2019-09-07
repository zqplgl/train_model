import os
import shutil
import pickle
import numpy as np
import cv2

def loadMat(file_path):
    fs = cv2.FileStorage(file_path,cv2.FILE_STORAGE_READ)
    im = fs.getNode("mat").mat()
    fs.release()
    return im

def classify():
    features = pickle.load(open("./features1.pkl", "rb"))
    label = loadMat("./labels.json.gz")
    centers = loadMat("./centers.json.gz")
    base_path = [line.strip() for line in open("./pic_path.txt").readlines()]
    result = {}

    folder_save_dir = "/media/zqp/data/train_data/patent_retrieval1/"
    for index in range(label.shape[0]):
        key = label[index][0]
        if key not in result.keys():
            result[key] = []
        result[key].append((base_path[index], index))

    for key in result.keys():
        pic_save_dir = folder_save_dir + "%06d/"%key

        if not os.path.exists(pic_save_dir):
            os.makedirs(pic_save_dir)
        for pic_path, index in result[key]:
            pic_name = os.path.basename(pic_path)

            cos = np.dot(features[index], centers[key])
            if cos < 0.8:
                continue
            shutil.move(pic_path, pic_save_dir+pic_name)
            print (pic_path)
            print (pic_save_dir+pic_name)

def cluster():
    centers = loadMat("./centers.json.gz")
    cluster_result = {}

    min_angle = 0
    for i in range(len(centers)-1):
        for j in range(i+1,len(centers)):
            angle = np.dot(centers[i], centers[j])

            if angle>0.9:
                if i in cluster_result.keys():
                    cluster_result[i].append(j)
                else:
                    flag = False
                    for key in cluster_result.keys():
                        if i in cluster_result[key]:
                            cluster_result[key].append(j)
                            flag = True
                            break

                    if not flag:
                        cluster_result[i] = [j]

                print(i,"**********",j, angle)
            if min_angle < angle:
                min_angle = angle

    print("max cos: ", min_angle)

    folder_dir = "/media/zqp/data/train_data/patent_retrieval1"
    for key in cluster_result.keys():
        for folder_name in cluster_result[key]:
            cmd = "mv %s/%06d/*.jpg %s/%06d/"%(folder_dir,folder_name,folder_dir,key)

            print(cmd)
            os.system(cmd)

def run():
    folder_dir = "/media/zqp/data/train_data/patent_retrieval1/"
    folder_names = sorted(os.listdir(folder_dir))

    folder_index = 0
    for folder_name in folder_names:
        print(folder_name, len(os.listdir(folder_dir+folder_name)))
        pic_dir = folder_dir+folder_name

        if len(os.listdir(pic_dir)) < 1:
            os.rmdir(pic_dir)
            continue

        if folder_name != "%06d"%folder_index:
            os.rename(folder_dir+folder_name, folder_dir+"%06d"%folder_index)
        folder_index += 1

if __name__=="__main__":
    classify()
    cluster()
    run()
