#!/usr/bin/env sh

width=256
height=256
folder_dir=/media/zqp/data/train_data/patent_retrieval/
echo "Create train lmdb.."
rm -rf img_train_lmdb
convert_imageset \
--shuffle \
--resize_width=${width} \
--resize_height=${height} \
--gray \
${folder_dir} \
train.txt \
img_train_lmdb

#echo "Create test lmdb.."
#rm -rf img_test_lmdb
#convert_imageset \
#--shuffle \
#--resize_width=${width} \
#--resize_height=${height} \
#${folderDir} \
#test.txt \
#img_test_lmdb

echo "All Done.."
