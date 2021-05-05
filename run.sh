#!/bin/bash

iters=200
ckpt_path="chkpt"

CUDA_VISIBLE_DEVICES=6 python train.py -c config/config.json --use-cuda --ckpt_path=${ckpt_path} --iters ${iters}
