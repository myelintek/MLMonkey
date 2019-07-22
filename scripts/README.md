# ImageNet dataset preprocessing

## download following ImageNet dataset
* ILSVRC2012_img_train.tar 138GB
* ILSVRC2012_img_val.tar 6.3GB

store imagenet dataset in folder ~/ILSVRC2012

## preprocessing script
refer to this link https://gluon-cv.mxnet.io/build/examples_datasets/recordio.html#imagerecord-file-for-imagenet

download imagenet.py (https://gluon-cv.mxnet.io/_downloads/8d1f4e8bf11292b2ca934f18039800ef/imagenet.py) and imagenet_val_maps.pklz (https://gluon-cv.mxnet.io/_downloads/e54c7538b26a1e6b16f6f9b85fc84654/imagenet_val_maps.pklz)

execute following command to generate recordio dataset
```
python imagenet.py --download-dir ~/ILSVRC2012 --target-dir ~/imagenet --with-rec
```
