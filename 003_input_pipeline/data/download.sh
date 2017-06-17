#!/usr/bin/env bash
if [ ! -d "COCO_samples/" ]; then
  wget http://cvlab.postech.ac.kr/~wgchang/data/others/COCO_samples.zip
  unzip COCO_samples.zip
  rm COCO_samples.zip
fi