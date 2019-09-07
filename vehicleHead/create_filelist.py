import os
import random
picRootDir = r'/media/zqp/data/dataSet/vehicleHeadMerge/'
train = open('train.txt','w')
test = open('test.txt','w')
labelMap = open("labelMap.txt","w")
num = 0
for folderName in sorted(os.listdir(picRootDir)):
    labelMap.write("%s\n"%folderName)
    picNames = [picName for picName in os.listdir(picRootDir+"/"+folderName)]
    picNum = len(picNames)
    random.shuffle(picNames)
    count = 0
    for picName in picNames:
        if count<float(picNum)*4/5:
            train.write("%s/%s %s\n"%(folderName,picName,num))
            count+=1
        else:
            test.write("%s/%s %s\n"%(folderName,picName,num))

    num += 1
    
