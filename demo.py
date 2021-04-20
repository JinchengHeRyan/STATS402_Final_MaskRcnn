# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import torch
import matplotlib.pyplot as plt
import pytorch_mask_rcnn as pmr
from torchvision import transforms
from PIL import Image


use_cuda = True
dataset = "coco"
# ckpt_path = "../ckpt/maskrcnn_voc-5.pth"
ckpt_path = "./chkpt/maskrcnn_coco-100.pth"
# data_dir = "E:/PyTorch/data/voc2012/"
data_dir = "/mingback/students/jincheng/data/COCO2017"

device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print("\ndevice: {}".format(device))

ds = pmr.datasets(dataset, data_dir, "val2017", train=True)
indices = torch.randperm(len(ds)).tolist()
d = torch.utils.data.Subset(ds, indices)

model = pmr.maskrcnn_resnet50(True, len(ds.classes) + 1).to(device)
model.eval()
model.head.score_thresh = 0.3

if ckpt_path:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(checkpoint["eval_info"])
    del checkpoint
    torch.cuda.empty_cache()
    
for p in model.parameters():
    p.requires_grad_(False)

# %%
iters = 30

for i, (image, target) in enumerate(d):
    image = image.to(device)
    target = {k: v.to(device) for k, v in target.items()}
    
    with torch.no_grad():
        result = model(image)
        
    plt.figure(figsize=(12, 15))
    pmr.show(image, result, ds.classes)

    if i >= iters - 1:
        break

# %%
img_dir = "image/IMG_3437.jpeg"
image = Image.open(img_dir)
image = image.convert("RGB")
image = transforms.ToTensor()(image)
image = image.to(device)
target = {k: v.to(device) for k, v in target.items()}

with torch.no_grad():
    result = model(image)

plt.figure(figsize=(12, 15))
pmr.show(image, result, ds.classes)

# %%
