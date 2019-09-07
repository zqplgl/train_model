import cv2
import pickle
import math
import numpy as np
from sklearn.cluster import KMeans
import os
import shutil

def kmeanclsuter():
    features = pickle.load(open("./features1.pkl","rb"))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret,label,center=cv2.kmeans(features,7000,None,criteria,100,cv2.KMEANS_PP_CENTERS)
    
    file1 = open("./label1.pkl", "wb")
    file2 = open("./center1.pkl", "wb")
    pickle.dump(label, file1)
    pickle.dump(center, file2)

def classifyView():
    label = pickle.load(open("./label1.pkl", "rb"))
    base_path = [line.strip() for line in open("./pic_path.txt").readlines()]
    result = {}
    test1 = []
    for index in range(label.shape[0]):
        key = label[index][0]
        if key not in result.keys():
            result[key] = []
        result[key].append(base_path[index])
        test1.append(base_path[index])

    print(len(test1))
    print(len(set(test1)))

    cv2.namedWindow("im",0)
    for key in result:
        flag = False
        for pic_path in result[key]:
            print(pic_path+"*************",key)
            im = cv2.imread(pic_path)
            cv2.imshow("im", im)
            if cv2.waitKey(0)==27:
                flag = True
                break
        if flag:
            break

def loadMat(file_path):
    fs = cv2.FileStorage(file_path,cv2.FILE_STORAGE_READ)
    im = fs.getNode("mat").mat()
    fs.release()
    return im

def classifyView1():
    label = loadMat("./labels.json.gz")
    centers = loadMat("./centers.json.gz")
    features = pickle.load(open("./features1.pkl","rb"))
    base_path = [line.strip() for line in open("./pic_path.txt").readlines()]
    result = {}
    for index in range(label.shape[0]):
        key = label[index][0]
        if key not in result.keys():
            result[key] = []
        result[key].append((base_path[index], index))

    count = 0
    for key in result:
        flag = False
        center = centers[key]

        for pic_path, index in result[key]:
            angle = np.dot(features[index], center)
            # angles = np.matmul(centers,features[index])
            # min_index = np.argsort(angles)[-1]
            if angle <0.8: continue
            print(angle)
            count+=1

    print("count: ",count)

    min_angle = 0
    for i in range(len(centers)-1):
        for j in range(i+1,len(centers)):
            angle = np.dot(centers[i], centers[j])

            if angle>0.85:
                print(i,"**********",j, angle)
            if min_angle < angle:
                min_angle = angle

    print("max cos: ", min_angle)

def classify():
    label = pickle.load(open("./label1.pkl", "rb"))
    base_path = [line.strip() for line in open("./pic_path.txt").readlines()]
    result = {}
    test1 = []

    folder_save_dir = "/media/zqp/data/train_data/patent_retrieval1/"
    for index in range(label.shape[0]):
        key = label[index][0]
        if key not in result.keys():
            result[key] = []
        result[key].append(base_path[index])
        test1.append(base_path[index])

    print(len(test1))
    print(len(set(test1)))

    for key in result:
        flag = False
        pic_save_dir = folder_save_dir + "%06d/"%key

        if not os.path.exists(pic_save_dir):
            os.makedirs(pic_save_dir)
        for pic_path in result[key]:
            pic_name = os.path.basename(pic_path)

            shutil.move(pic_path, pic_save_dir+pic_name)
            print (pic_path)
            print (pic_save_dir+pic_name)


if __name__=="__main__":
    # kmeanclsuter()
    # classifyView()
    classifyView1()
    # classify()
