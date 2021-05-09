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
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

use_cuda = True
# dataset = "coco"
dataset = "voc"
# ckpt_path = "../ckpt/maskrcnn_voc-5.pth"
ckpt_path = "./chkpt/saved/maskrcnn_voc-306.pth"
# data_dir = "E:/PyTorch/data/voc2012/"
# data_dir = "/mingback/students/jincheng/data/COCO2017"
data_dir = "/mingback/students/jincheng/data/VOC2012/VOCdevkit/VOC2012"

device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print("\ndevice: {}".format(device))

ds = pmr.datasets(
    dataset, data_dir, "val2017" if dataset == "coco" else "val", train=True
)
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
iters = 3

for i, (image, target) in enumerate(d):
    image = image.to(device)
    target = {k: v.to(device) for k, v in target.items()}

    with torch.no_grad():
        result = model(image)

    plt.figure(figsize=(12, 15))
    pmr.show(image, result, ds.classes)
    plt.imshow(torch.sum(result["masks"], axis=0).cpu())
    plt.axis("off")
    plt.show()

    if i >= iters - 1:
        break

# %%
img_dir_list = ["image/IMG_0952.jpeg", "image/IMG_3309.jpeg", "image/IMG_3454.jpeg"]
for img_dir in img_dir_list:
    image = Image.open(img_dir)
    image = image.convert("RGB")
    image = transforms.ToTensor()(image)
    image = image.to(device)
    target = {k: v.to(device) for k, v in target.items()}

    with torch.no_grad():
        result = model(image)

    plt.figure(figsize=(12, 15))
    print(result.keys())
    pmr.show(image, result, ds.classes)
    plt.imshow(torch.sum(result["masks"], axis=0).cpu())
    plt.axis("off")
    plt.show()

# %%
# High resolution input
img_dir_list = ["image/mmexport1535710463698.jpg"]
for img_dir in img_dir_list:
    image = Image.open(img_dir)
    image = image.convert("RGB")
    image = transforms.ToTensor()(image)

    target = {k: v.to(device) for k, v in target.items()}

    n_range = range(2, 10)

    obj_count = list()

    print(image.shape)
    for n in n_range:
        mask_all = torch.zeros(n * image.shape[1] // n, n * image.shape[2] // n)
        print("current n = ", n)
        ans = 0
        for i in range(n):
            for j in range(n):
                img_cut = image[
                    :,
                    i * image.shape[1] // n : (i + 1) * image.shape[1] // n,
                    j * image.shape[2] // n : (j + 1) * image.shape[2] // n,
                ]
                img_cut = img_cut.to(device)
                with torch.no_grad():
                    result = model(img_cut)

                ans += result["masks"].shape[0]
                mask_all[
                    i * image.shape[1] // n : (i + 1) * image.shape[1] // n,
                    j * image.shape[2] // n : (j + 1) * image.shape[2] // n,
                ] = torch.sum(result["masks"], axis=0)

        obj_count.append(ans)
        plt.figure(figsize=(12, 15))
        plt.imshow(mask_all)
        plt.axis("off")
        plt.show()
    print(obj_count)

# %%
plot_x = [i ** 2 for i in n_range]
print(plot_x)

plt.figure(figsize=(12, 8))
plt.plot(plot_x, obj_count, "o-")
plt.xlabel("Number of blocks cut", fontsize=15)
plt.ylabel("Detected objects count", fontsize=15)
plt.title("High resolution input strategy", fontsize=20)
plt.show()

# %%
img_dir_list = ["image/IMG_0952.jpeg"]
for img_dir in img_dir_list:
    image = Image.open(img_dir)
    image = image.convert("RGB")
    image_gray = image.convert("L")
    image_gray = transforms.ToTensor()(image_gray)
    image = transforms.ToTensor()(image)
    image = image.to(device)
    target = {k: v.to(device) for k, v in target.items()}

    with torch.no_grad():
        result = model(image)

    plt.figure(figsize=(12, 15))
    thre = 0.6
    img_mani = torch.zeros(image.shape).cuda()
    img_mani[:, torch.sum(result["masks"], axis=0) > thre] = image[
        :, torch.sum(result["masks"], axis=0) > thre
    ]
    img_mani[:, torch.sum(result["masks"], axis=0) <= thre] = (
        0.2989 * image[0, torch.sum(result["masks"], axis=0) <= thre]
        + 0.5870 * image[1, torch.sum(result["masks"], axis=0) <= thre]
        + 0.1140 * image[2, torch.sum(result["masks"], axis=0) <= thre]
    )
    plt.imshow(img_mani.permute(1, 2, 0).cpu())
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(12, 15))

    pmr.show(image, result, ds.classes)
    plt.imshow(torch.sum(result["masks"], axis=0).cpu())
    plt.axis("off")
    plt.show()
