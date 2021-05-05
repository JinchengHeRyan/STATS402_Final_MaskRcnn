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
import os
import time
import torch
import pytorch_mask_rcnn as pmr

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda":
        pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
    
    d_test = pmr.datasets(args.dataset, args.data_dir, "val2017", train=True) # set train=True for eval

    print(args)
    num_classes = len(d_test.classes) + 1
    model = pmr.maskrcnn_resnet50(False, num_classes).to(device)
    
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(checkpoint["eval_info"])
    del checkpoint
    torch.cuda.empty_cache()

    print("evaluating only...")
    B = time.time()
    eval_output, iter_eval = pmr.evaluate(model, d_test, device, args)
    B = time.time() - B
    print(eval_output)
    print("\ntotal time of this evaluation: {:.2f} s, speed: {:.2f} FPS".format(B, args.batch_size / iter_eval))
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--data-dir")
    parser.add_argument("--iters", type=int, default=-1)
    
    args = parser.parse_args([]) # for Jupyter Notebook
    
    args.use_cuda = True
    args.data_dir = "/mingback/students/jincheng/data/COCO2017"
    args.ckpt_path = "chkpt/saved/maskrcnn_coco-1000.pth"
    args.results = os.path.join(os.path.dirname(args.ckpt_path), "results.pth")
    
    main(args)
    
    

# %%
