#!/usr/bin/env bash
# download COCO dataset
wget http://cvlab.postech.ac.kr/~wgchang/data/others/COCO_samples.zip
if [!-d "data"]
then
  mkdir data
fi
unzip COCO_samples -d data/
rm COCO_samples.zip
