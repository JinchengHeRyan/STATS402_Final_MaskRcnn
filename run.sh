#!/bin/bash

dataset="coco"
epochs=100
iters=200
ckpt_path="chkpt"

if [ $dataset = "voc" ]; then
  data_dir="/data/voc2012/VOCdevkit/VOC2012/"
elif [ $dataset = "coco" ]; then
  data_dir="/mingback/students/jincheng/data/COCO2017"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --use-cuda --epochs ${epochs} --iters ${iters} --dataset ${dataset} --data-dir ${data_dir} --ckpt_path=${ckpt_path}
