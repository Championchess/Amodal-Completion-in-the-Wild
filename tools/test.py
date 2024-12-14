import argparse
import yaml
import os
import json
import numpy as np
from PIL import Image
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys

sys.path.append(".")
from libs.datasets import reader
from libs import models
import inference as infer
from libs import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

import torchvision.transforms as transforms

import ipdb
from skimage import measure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--load-model", required=True, type=str)
    parser.add_argument("--order-method", required=True, type=str)
    parser.add_argument("--amodal-method", required=True, type=str)
    parser.add_argument("--order-th", default=0.1, type=float)
    parser.add_argument("--amodal-th", default=0.2, type=float)
    parser.add_argument("--annotation", required=True, type=str)
    parser.add_argument("--image-root", required=True, type=str)
    parser.add_argument("--test-num", default=-1, type=int)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--dilate_kernel", default=0, type=int)
    args = parser.parse_args()
    return args


def main(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in config.items():
        setattr(args, k, v)

    if not hasattr(args, "exp_path"):
        args.exp_path = os.path.dirname(args.config)

    tester = Tester(args)
    tester.run()


class Tester(object):
    def __init__(self, args):
        self.args = args
        self.prepare_data()

    def prepare_data(self):
        config = self.args.data
        dataset = "COCOA"
        self.data_root = self.args.image_root
        if dataset == "COCOA":
            self.data_reader = reader.COCOADataset(self.args.annotation)
        self.data_length = self.data_reader.get_image_length()
        self.dataset = dataset
        if self.args.test_num != -1:
            self.data_length = self.args.test_num

    def prepare_model(self):
        self.model = models.__dict__[self.args.model["algo"]](
            self.args.model, dist_model=False
        )
        self.model.load_state(self.args.load_model)
        self.model.switch_to("eval")

    def expand_bbox(self, bboxes):
        new_bboxes = []
        for bbox in bboxes:
            centerx = bbox[0] + bbox[2] / 2.0
            centery = bbox[1] + bbox[3] / 2.0
            size = max(
                [
                    np.sqrt(bbox[2] * bbox[3] * self.args.data["enlarge_box"]),
                    bbox[2] * 1.1,
                    bbox[3] * 1.1,
                ]
            )
            new_bbox = [
                int(centerx - size / 2.0),
                int(centery - size / 2.0),
                int(size),
                int(size),
            ]
            new_bboxes.append(new_bbox)
        return np.array(new_bboxes)

    def run(self):
        self.prepare_model()
        self.infer()

    def infer(self):
        order_th = self.args.order_th
        amodal_th = self.args.amodal_th

        self.args.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    self.args.data["data_mean"], self.args.data["data_std"]
                ),
            ]
        )

        segm_json_results = []
        self.count = 0

        allpair_true_rec = utils.AverageMeter()
        allpair_rec = utils.AverageMeter()
        occpair_true_rec = utils.AverageMeter()
        occpair_rec = utils.AverageMeter()
        intersection_rec = utils.AverageMeter()
        union_rec = utils.AverageMeter()
        target_rec = utils.AverageMeter()

        inv_intersection_rec = utils.AverageMeter()
        inv_union_rec = utils.AverageMeter()

        list_acc, list_iou = [], []
        list_inv_iou = []

        print(self.data_length)

        for i in range(0, self.data_length):
            print(i)
            modal, category, bboxes, amodal_gt, image_fn, occ_l = (
                self.data_reader.get_image_instances(
                    i, with_gt=True, load_occ_label=True
                )
            )

            # data
            image = Image.open(os.path.join(self.data_root, image_fn)).convert(
                "RGB"
            )
            if (
                image.size[0] != modal.shape[2]
                or image.size[1] != modal.shape[1]
            ):
                image = image.resize((modal.shape[2], modal.shape[1]))

            image = np.array(image)
            h, w = image.shape[:2]
            bboxes = self.expand_bbox(bboxes)

            if self.args.order_method == "aw":
                pass
            else:
                raise Exception(
                    "No such order method: {}".format(self.args.order_method)
                )

            if self.args.amodal_method == "aw_sdm5":  # supervised
                amodal_patches_pred = infer.infer_amodal_aw_sdm(
                    self.model,
                    image_fn,
                    modal,
                    category,
                    bboxes,
                    use_rgb=self.args.model["use_rgb"],
                    th=amodal_th,
                    input_size=512,
                    min_input_size=16,
                    interp="nearest",
                    args=args,
                )
                amodal_pred = infer.patch_to_fullimage(
                    amodal_patches_pred, bboxes, h, w, interp="linear"
                )

            intersection = ((amodal_pred == 1) & (amodal_gt == 1)).sum()
            union = ((amodal_pred == 1) | (amodal_gt == 1)).sum()
            target = (amodal_gt == 1).sum()
            intersection_rec.update(intersection)
            union_rec.update(union)
            target_rec.update(target)

            # for invisible mIoU
            inv_intersection = (
                (amodal_pred == 1) & (amodal_gt == 1) & (modal == 0)
            ).sum()
            inv_union = (
                ((amodal_pred == 1) | (amodal_gt == 1)) & (modal == 0)
            ).sum()

            inv_intersection_rec.update(inv_intersection)
            inv_union_rec.update(inv_union)

            # for computing p-score
            # list_acc.append(occpair_true/(occpair+1e-6))
            list_iou.append(intersection / (union + 1e-6))
            list_inv_iou.append(inv_intersection / (inv_union + 1e-6))

        miou = intersection_rec.sum / (union_rec.sum + 1e-10)  # mIoU
        pacc = intersection_rec.sum / (target_rec.sum + 1e-10)  # pixel accuracy

        inv_miou = inv_intersection_rec.sum / (
            inv_union_rec.sum + 1e-10
        )  # mIoU

        print(
            "Evaluation results.  \
              mIoU: {:.5g}, pAcc: {:.5g}, inv_mIoU: {:.5g}".format(
                miou, pacc, inv_miou
            )
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
