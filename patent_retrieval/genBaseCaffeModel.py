import caffe

def run():
    prototxt = "./pretrain.prototxt"
    weightfile = "./final.caffemodel"
    net = caffe.Net(prototxt,weightfile,caffe.TEST)

    net.save("./pretrain.caffemodel")

if __name__=="__main__":
    run()
