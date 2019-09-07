import os
import random
def run():
    pic_root_dir = r'/media/zqp/data/train_data/patent_retrieval'
    train = open('train.txt','w')
    test = open('test.txt','w')
    label = open("label.txt","w")
    num = 0
    for folder_name in sorted(os.listdir(pic_root_dir)):
        label.write("%s\n"%folder_name)
        pic_names = [pic_name for pic_name in os.listdir(pic_root_dir+"/"+folder_name)]
        pic_num = len(pic_names)
        random.shuffle(pic_names)
        for pic_name in pic_names:
            train.write("%s/%s %s\n"%(folder_name,pic_name,num))

        num += 1

if __name__=="__main__":
    run()
    
