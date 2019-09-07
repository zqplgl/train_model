train: create_filelist.py solver.prototxt train_val.prototxt create_lmdb.sh
	python create_filelist.py
	./create_lmdb.sh
	compute_image_mean img_train_lmdb mean.binaryproto
	caffe train --solver=solver.prototxt \
	--weights VGG_FACE.caffemodel \
	2>&1 | tee log.txt

clean:
	-rm -rf img_* test.txt train.txt model_iter_* labelMap.txt log.txt mean.binaryproto


