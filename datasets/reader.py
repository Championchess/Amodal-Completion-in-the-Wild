import numpy as np
import sys
sys.path.append('.')

import cvbase as cvb
import pycocotools.mask as maskUtils
import utils
import mmcv
import ipdb



def read_COCOA(ann, h, w, load_occ_label=False):
    if 'visible_mask' in ann.keys(): # occluded
        rle = [ann['visible_mask']]
        if load_occ_label:
            occ_l = 0
        else:
            occ_l = -1
    else: # not occluded
        rles = maskUtils.frPyObjects([ann['segmentation']], h, w)
        rle = maskUtils.merge(rles)
        if load_occ_label:
            occ_l = 1
        else:
            occ_l = -1
    modal = maskUtils.decode(rle).squeeze()
    if np.all(modal != 1):
        # if the object if fully occluded by others,
        # use amodal bbox as an approximated location,
        # note that it will produce random amodal results.
        amodal = maskUtils.decode(maskUtils.merge(
            maskUtils.frPyObjects([ann['segmentation']], h, w)))
        bbox = utils.mask_to_bbox(amodal)
    else:
        bbox = utils.mask_to_bbox(modal)
    return modal, bbox, 1, occ_l # category as constant 1


class COCOADataset(object):

    def __init__(self, annot_fn):
        data = cvb.load(annot_fn)
        self.images_info = data['images']
        self.annot_info = data['annotations']

        self.indexing = []
        for i, ann in enumerate(self.annot_info):
            for j in range(len(ann['regions'])):
                self.indexing.append((i, j))

    def get_instance_length(self):
        return len(self.indexing)

    def get_image_length(self):
        return len(self.images_info)

    def get_gt_ordering(self, imgidx):
        num = len(self.annot_info[imgidx]['regions'])
        gt_order_matrix = np.zeros((num, num), dtype=np.int)
        order_str = self.annot_info[imgidx]['depth_constraint']
        if len(order_str) == 0:
            return gt_order_matrix
        order_str = order_str.split(',')
        for o in order_str:
            idx1, idx2 = o.split('-')
            idx1, idx2 = int(idx1) - 1, int(idx2) - 1
            gt_order_matrix[idx1, idx2] = 1
            gt_order_matrix[idx2, idx1] = -1
        return gt_order_matrix # num x num

    def get_instance(self, idx, with_gt=False, load_occ_label=False):
        imgidx, regidx = self.indexing[idx]
        # img
        img_info = self.images_info[imgidx]
        image_fn = img_info['file_name']
        w, h = img_info['width'], img_info['height']
        # region
        reg = self.annot_info[imgidx]['regions'][regidx]
        modal, bbox, category, occ_l = read_COCOA(reg, h, w, load_occ_label=load_occ_label)
        if with_gt:
            amodal = maskUtils.decode(maskUtils.merge(
                maskUtils.frPyObjects([reg['segmentation']], h, w)))
        else:
            amodal = None
        return modal, bbox, category, image_fn, amodal, occ_l

    def get_image_instances(self, idx, with_gt=False, with_anns=False, ignore_stuff=False, load_occ_label=False):
        ann_info = self.annot_info[idx]
        img_info = self.images_info[idx]
        image_fn = img_info['file_name']
        w, h = img_info['width'], img_info['height']
        ret_modal = []
        ret_bboxes = []
        ret_category = []
        ret_amodal = []
        ret_occ_l = []
        for reg in ann_info['regions']:
            if ignore_stuff and reg['isStuff']:
                continue
            modal, bbox, category, occ_l = read_COCOA(reg, h, w, load_occ_label=load_occ_label)
            ret_modal.append(modal)
            ret_bboxes.append(bbox)
            ret_category.append(category)
            ret_occ_l.append(occ_l)
            if with_gt:
                amodal = maskUtils.decode(maskUtils.merge(
                    maskUtils.frPyObjects([reg['segmentation']], h, w)))
                ret_amodal.append(amodal)
        if with_anns:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn, ann_info
        elif load_occ_label:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn, np.array(ret_occ_l)
        else:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn



def read_MP3D(ann, h, w, load_occ_label=False):
    assert 'visible_mask' in ann.keys() # must occluded
    m_rle = [ann['visible_mask']]
    modal = maskUtils.decode(m_rle).squeeze()
    a_rle = [ann['segmentation']]
    amodal = maskUtils.decode(a_rle).squeeze()
    bbox = utils.mask_to_bbox(modal)
    category = ann['category_id']
    return modal, bbox, category, amodal # category as constant 1


class MP3DDataset(object):

    def __init__(self, annot_fn):
        data = cvb.load(annot_fn)
        self.images_info = data['images']
        self.annot_info = data['annotations']

        self.indexing = []
        for i, ann in enumerate(self.annot_info):
            for j in range(len(ann['regions'])):
                self.indexing.append((i, j))

    def get_instance_length(self):
        return len(self.indexing)

    def get_image_length(self):
        return len(self.images_info)

    def get_gt_ordering(self, imgidx):
        num = len(self.annot_info[imgidx]['regions'])
        gt_order_matrix = np.zeros((num, num), dtype=np.int)
        order_str = self.annot_info[imgidx]['depth_constraint']
        if len(order_str) == 0:
            return gt_order_matrix
        order_str = order_str.split(',')
        for o in order_str:
            idx1, idx2 = o.split('-')
            idx1, idx2 = int(idx1) - 1, int(idx2) - 1
            gt_order_matrix[idx1, idx2] = 1
            gt_order_matrix[idx2, idx1] = -1
        return gt_order_matrix # num x num

    def get_instance(self, idx, with_gt=False, load_occ_label=False):
        imgidx, regidx = self.indexing[idx]
        # img
        img_info = self.images_info[imgidx]
        image_fn = img_info['file_name']
        w, h = img_info['width'], img_info['height']
        # region
        reg = self.annot_info[imgidx]['regions'][regidx]
        modal, bbox, category, amodal = read_MP3D(reg, h, w, load_occ_label=load_occ_label)
        return modal, bbox, category, image_fn, amodal

    def get_image_instances(self, idx, with_gt=False, with_anns=False, ignore_stuff=False, load_occ_label=False):
        ann_info = self.annot_info[idx]
        img_info = self.images_info[idx]
        image_fn = img_info['file_name']
        w, h = img_info['width'], img_info['height']
        ret_modal = []
        ret_bboxes = []
        ret_category = []
        ret_amodal = []
        for reg in ann_info['regions']:
            if ignore_stuff and reg['isStuff']:
                continue
            modal, bbox, category, amodal = read_MP3D(reg, h, w, load_occ_label=load_occ_label)
            ret_modal.append(modal)
            ret_bboxes.append(bbox)
            ret_category.append(category)
            ret_amodal.append(amodal)
        if with_anns:
            return np.array(ret_modal), ret_category, np.array(ret_bboxes), np.array(ret_amodal), image_fn, ann_info
        else:
            return np.array(ret_modal), ret_category, np.array(ret_bboxes), np.array(ret_amodal), image_fn