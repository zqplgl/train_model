#!/usr/bin/env sh

width=256
height=256
folderDir=/media/zqp/data/dataSet/vehicleHeadMerge/
echo "Create train lmdb.."
rm -rf img_train_lmdb
convert_imageset \
--shuffle \
--resize_width=${width} \
--resize_height=${height} \
${folderDir} \
train.txt \
img_train_lmdb

echo "Create test lmdb.."
rm -rf img_test_lmdb
convert_imageset \
--shuffle \
--resize_width=${width} \
--resize_height=${height} \
${folderDir} \
test.txt \
img_test_lmdb

echo "All Done.."
