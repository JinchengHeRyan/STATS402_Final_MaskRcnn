import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os, sys

sys.path.append(os.path.abspath(os.pardir))

import pytorch_mask_rcnn as pmr

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

use_cuda = True
ckpt_path = "../chkpt/saved/maskrcnn_coco-1000.pth"
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print("\ndevice: {}".format(device))

dataset = "coco"
data_dir = "/mingback/students/jincheng/data/COCO2017"
ds = pmr.datasets(dataset, data_dir, "val2017", train=True)
model = pmr.maskrcnn_resnet50(True, len(ds.classes) + 1).to(device)
model.eval()
model.head.score_thresh = 0.85

if ckpt_path:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(checkpoint["eval_info"])
    del checkpoint
    torch.cuda.empty_cache()

for p in model.parameters():
    p.requires_grad_(False)

img_dir_list = ["./img/{}.jpg".format(i) for i in range(1, 695)]
for idx in range(len(img_dir_list)):
    img_dir = img_dir_list[idx]
    print("current img: {}".format(img_dir))
    image = Image.open(img_dir)
    image = image.convert("RGB")
    image = transforms.ToTensor()(image)
    image = image.to(device)
    # target = {k: v.to(device) for k, v in target.items()}

    with torch.no_grad():
        result = model(image)

    plt.figure(figsize=(12, 8))
    savePath = "./output/{}.jpg".format(idx)
    maskSavePath = "./output/{}_mask.jpg".format(idx)
    pmr.show(image, result, ds.classes, save_path=savePath, figsize=(12, 8))
    # plt.imshow(torch.sum(result["masks"], axis=0).cpu())
    # plt.savefig(maskSavePath)
    # plt.show()
