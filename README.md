# Pixel-wise segmentation is all you need

### Before running the code

The required packages are as the following, the file `requirements.txt` also contains these information. This repo can
work with these requirements at least on our computer with Linux system equipped GeForce GTX 1080 Ti (10G).

```
matplotlib==3.1.1
numpy==1.17.2
opencv_python==4.5.1.48
torchvision==0.8.2
torch==1.7.1
pycocotools==2.0.2
Pillow==8.2.0
```

Before run the code, first should download the following data:

VOC2012:

http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

We also put VOC dataset in the Duke Box, which can be downloaded from there.

COCO2017:

http://images.cocodataset.org/zips/train2017.zip

http://images.cocodataset.org/annotations/annotations_trainval2017.zip

http://images.cocodataset.org/zips/val2017.zip

http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

http://images.cocodataset.org/zips/test2017.zip

http://images.cocodataset.org/annotations/image_info_test2017.zip

All the above links of COCO dataset should be downloaded and unzip correctly. Since it is too big for us to upload, so
it can only be downloaded by yourself.

Then edit `config/config.json`, if the dataset is `COCO2017`, then should edit this file in this way

```json
{
    "epochs": 1100,
    "dataset": "coco",
    "data_dir": "/your/path/to/data/COCO2017"
}
```

If the dataset is `VOC2012`, then should edit this config file in this way

```json
{
    "epochs": 1100,
    "dataset": "voc",
    "data_dir": "/your/path/to/data/VOCdevkit/VOC2012"
}
```

### Train the model

After editing the config file, then edit `run.sh`, based on your own gpu situation, for example, if you have one gpu,
then should edit the first part of the last line in `run.sh` to be like this

```shell
CUDA_VISIBLE_DEVICES=0 python train.py -c config/config.json --use-cuda --ckpt_path=${ckpt_path} --iters ${iters}
```

After editing the `run.sh`, run the following command to begin training

```shell
bash run.sh
```

During the training, the saved parameters of the model would be saved in `chkpt/` directory, and the log file would be
saved in `logs/` directory.

### Run the demo code

As you can see in this repo, there are files as `demo.ipynb`, `demo.py`, `eval.ipynb`, `eval.py`. The code content
between `demo.ipynb` and `demo.py`, and between `eval.ipynb` and `eval.py` are identical, the reason to store `demo.py`
and `eval.py` is only for convenient git commit, I suggest if you want to see the demo, run the notebook `demo.ipynb`.
And we have already ran this notebook, you can directly see the result of the demo, and of course you can run again this
by yourselves but with at least one GPU. In this notebook, it contains the demo of doing segmentation on our example
pictures, the demo of high resolution input image, and the demo of extracting the object with RGB color scale and all
the backgound is gray scale.

Also before running the demo notebook, change the following line of code in the first cell of the notebook based on your
own GPU situation

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

This demo notebook should load the learned model parameters in `chkpt/saved/` directory, and our demo notebook loads the
parameters learned from VOC2012 dataset, if you want to change the model learned from COCO dataset, the three
parameters `dataset`, `ckpt_path`, `data_dir` need to change to the correct version.

Another important thing is that in order to successfully run the demo code, the dataset is needed to load some important
files. 
