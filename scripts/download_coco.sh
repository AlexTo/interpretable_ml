aria2c -d data/coco -x 4 --auto-file-renaming=false http://images.cocodataset.org/zips/train2014.zip
aria2c -d data/coco -x 4 --auto-file-renaming=false http://images.cocodataset.org/zips/val2014.zip
aria2c -d data/coco -x 4 --auto-file-renaming=false http://images.cocodataset.org/zips/test2014.zip
aria2c -d data/coco -x 4 --auto-file-renaming=false http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip -d data/coco -n data/coco/train2014.zip
unzip -d data/coco -n data/coco/val2014.zip
unzip -d data/coco -n data/coco/test2014.zip
unzip -d data/coco -n data/coco/annotations_trainval2014.zip