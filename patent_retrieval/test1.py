import numpy as np
import cv2
import pickle

def run():
    m = 5
    k = 8
    n = 2
    a = []
    for i in range(m*k):
        a.append(i*1.2)

    a = np.array(a).reshape((m,k))

    b = []
    for i in range(k*n):
        b.append(i*5.6)

    b = np.array(b).reshape((k,n))


    print(a)
    print()
    print(b)
    print()
    print(np.matmul(a,b))
    print()
    print(np.matmul(b.T,a.T))

def run_1():
    fs = cv2.FileStorage("/home/zqp/github/kmeans/bin/test.json.gz",cv2.FILE_STORAGE_READ)
    num = fs.getNode("num").real()
    height = fs.getNode("height").real()
    im = fs.getNode("im").mat()
    cv2.imshow("im",im)
    cv2.waitKey(0)

def run_2():
    features = pickle.load(open("./features1.pkl", "rb"))[:150000]
    fs = cv2.FileStorage("./features1.json.gz",cv2.FILE_STORAGE_WRITE)
    fs.write("mat", features)
    fs.release()

def run_3():
    a = np.array([[1,2,3,4]])
    b = cv2.norm(a)
    b = a/b
    b = np.dot(b[0],b[0])
    print(b)

def run_4():
    a = np.array([[1,2,3,4],[5,6,7,8]],dtype=np.float32)
    b = cv2.reduce(a,1,cv2.REDUCE_MIN,dtype=-1)
    print(b)




if __name__=="__main__":
    # run()
    # run_1()
    run_2()
    # run_4()
