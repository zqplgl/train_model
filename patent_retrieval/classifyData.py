import pickle
import shutil
import os
import numpy as np

def classify():
    features = pickle.load(open("./features1.pkl", "rb"))
    label = pickle.load(open("./label1.pkl", "rb"))
    centers = pickle.load(open("./center1.pkl", "rb"))
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

if __name__=="__main__":
    classify()
