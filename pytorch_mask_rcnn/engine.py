import time
import sys

import torch

from .utils import Meter, TextArea

try:
    from .datasets import CocoEvaluator, prepare_for_coco
except:
    pass


class AverageVal(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, optimizer, data_loader, device, epoch, args, logger=None):
    rpn_objectness_losses = AverageVal()
    rpn_box_losses = AverageVal()
    roi_classifier_losses = AverageVal()
    roi_box_losses = AverageVal()
    roi_mask_losses = AverageVal()

    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        S = time.time()

        losses = model(image, target)
        total_loss = sum(losses.values())
        m_m.update(time.time() - S)

        rpn_objectness_losses.update(losses["rpn_objectness_loss"].item())
        rpn_box_losses.update(losses["rpn_box_loss"].item())
        roi_classifier_losses.update(losses["roi_classifier_loss"].item())
        roi_box_losses.update(losses["roi_box_loss"].item())
        roi_mask_losses.update(losses["roi_mask_loss"].item())

        S = time.time()
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        b_m.update(time.time() - S)

        if num_iters % args.print_freq == 0:
            print(
                "{}\t".format(num_iters),
                "\t".join("{:.3f}".format(l.item()) for l in losses.values()),
            )

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break

    A = time.time() - A
    print(
        "iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(
            1000 * A / iters, 1000 * t_m.avg, 1000 * m_m.avg, 1000 * b_m.avg
        )
    )
    if logger is not None:
        logger.info(
            "[Train] Epoch:{}\t"
            "rpn_objectness_loss:{:.4f}\t"
            "rpn_box_loss:{:.4f}\t"
            "roi_classifier_loss:{:.4f}\t"
            "roi_box_loss:{:.4f}\t"
            "roi_mask_loss:{:.4f}".format(
                epoch,
                rpn_objectness_losses.avg,
                rpn_box_losses.avg,
                roi_classifier_losses.avg,
                roi_box_losses.avg,
                roi_mask_losses.avg,
            ),
        )
    return A / iters


def evaluate(model, data_loader, device, args, generate=True):
    if generate:
        iter_eval = generate_results(model, data_loader, device, args)

    dataset = data_loader  #
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(args.results, map_location="cpu")

    S = time.time()
    if len(results) > 0:
        coco_evaluator.accumulate(results)
    print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()

    if len(results) > 0:
        coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp

    return output, iter_eval


# generate results file
@torch.no_grad()
def generate_results(model, data_loader, device, args):
    iters = len(data_loader) if args.iters < 0 else args.iters
    ann_labels = data_loader.ann_labels

    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()

        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        torch.cuda.synchronize()
        output = model(image)
        m_m.update(time.time() - S)

        prediction = {
            target["image_id"].item(): {k: v.cpu() for k, v in output.items()}
        }
        coco_results.extend(prepare_for_coco(prediction, ann_labels))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break

    A = time.time() - A
    print(
        "iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(
            1000 * A / iters, 1000 * t_m.avg, 1000 * m_m.avg
        )
    )

    S = time.time()
    print("all gather: {:.1f}s".format(time.time() - S))
    torch.save(coco_results, args.results)

    return A / iters
