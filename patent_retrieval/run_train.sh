#./create_lmdb.sh
#cp -r ./img_train_lmdb img_test_lmdb
#caffe train --solver=solver.prototxt \
caffe train --solver=solver.prototxt --weights=./pretrain.caffemodel \
	2>&1 | tee log.txt
