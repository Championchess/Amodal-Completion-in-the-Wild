import numpy as np
import cv2

import torch
import torch.nn as nn

import utils
import pdb
from skimage.morphology import convex_hull
from torch.nn import functional as F
import matplotlib.pyplot as plt

import ipdb
from PIL import Image
import pycocotools.mask as maskUtils

def to_eraser(inst, bbox, newbbox):
    inst = inst.squeeze(0).numpy()
    final_h, final_w = inst.shape[:2]
    w, h = bbox.numpy()[2:]
    inst = cv2.resize(inst, (w, h), interpolation=cv2.INTER_LINEAR)
    offbbox = [newbbox[0] - bbox[0], newbbox[1] - bbox[1], newbbox[2], newbbox[3]]
    eraser = utils.crop_padding(inst, offbbox, pad_value=(0,))
    eraser = cv2.resize(eraser, (final_w, final_h), interpolation=cv2.INTER_NEAREST)
    #eraser = (eraser >= 0.5).astype(inst.dtype)
    return torch.from_numpy(eraser).unsqueeze(0)

def get_eraser(inst_ind, idx, bbox, input_size):
    inst_ind = inst_ind.numpy()
    bbox = bbox.numpy().tolist()
    eraser = cv2.resize(utils.crop_padding(inst_ind, bbox, pad_value=(0,)),
        (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    eraser = (eraser == idx + 1)
    return torch.from_numpy(eraser.astype(np.float32)).unsqueeze(0)



# add by guanqi - 10.23
def net_forward_aw_sdm(model, image, inmodal_patch, eraser, use_rgb, th, args=None, debug=False, no_eraser=False):
    if use_rgb:
        for layer_i in image.keys():
            image[layer_i] = torch.tensor(image[layer_i]).unsqueeze(0)
            image[layer_i] = image[layer_i].cuda()

    inmodal_patch = torch.from_numpy(inmodal_patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():        
        if use_rgb:
            output = model.model(torch.cat([inmodal_patch], dim=1), image)
        else:
            output = model.model(inmodal_patch)

    output.detach_()

    std = torch.zeros_like(output[0, 0])
    
    result = output.argmax(1)[0].cpu().numpy().astype(np.uint8)

    if debug:
        return result, std.cpu().numpy()
    else:
        return result

        
def recover_mask(mask, bbox, h, w, interp):
    size = bbox[2]
    if interp == 'linear':
        mask = (cv2.resize(mask.astype(np.float32), (size, size),
            interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    else:
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    woff, hoff = bbox[0], bbox[1]
    newbbox = [-woff, -hoff, w, h]
    return utils.crop_padding(mask, newbbox, pad_value=(0,))

def resize_mask(mask, size, interp):
    if interp == 'linear':
        return (cv2.resize(
            mask.astype(np.float32), (size, size),
            interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    else:
        return cv2.resize(
            mask, (size, size), interpolation=cv2.INTER_NEAREST)

def infer_amodal_hull(inmodal, bboxes, order_matrix, order_grounded=True):
    amodal = []
    num = inmodal.shape[0]
    for i in range(num):
        m = inmodal[i]
        hull = convex_hull.convex_hull_image(m).astype(np.uint8)
        if order_grounded:
            assert order_matrix is not None
            ancestors = get_ancestors(order_matrix, i)
            eraser = (inmodal[ancestors, ...].sum(axis=0) > 0).astype(np.uint8) # union
            hull[(eraser == 0) & (m == 0)] = 0
        amodal.append(hull)
    return amodal

def infer_order_hull(inmodal):
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=np.int)
    occ_value_matrix = np.zeros((num, num), dtype=np.float32)
    for i in range(num):
        for j in range(i + 1, num):
            if bordering(inmodal[i], inmodal[j]):
                amodal_i = convex_hull.convex_hull_image(inmodal[i])
                amodal_j = convex_hull.convex_hull_image(inmodal[j])
                occ_value_matrix[i, j] = ((amodal_i > inmodal[i]) & (inmodal[j] == 1)).sum()
                occ_value_matrix[j, i] = ((amodal_j > inmodal[j]) & (inmodal[i] == 1)).sum()
    order_matrix[occ_value_matrix > occ_value_matrix.transpose()] = -1
    order_matrix[occ_value_matrix < occ_value_matrix.transpose()] = 1
    order_matrix[(occ_value_matrix == 0) & (occ_value_matrix == 0).transpose()] = 0
    return order_matrix

def infer_order_area(inmodal, above='larger'):
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=np.int)
    for i in range(num):
        for j in range(i + 1, num):
            if bordering(inmodal[i], inmodal[j]):
                area_i = inmodal[i].sum()
                area_j = inmodal[j].sum()
                if (area_i < area_j and above == 'larger') or \
                   (area_i >= area_j and above == 'smaller'):
                    order_matrix[i, j] = -1 # i occluded by j
                    order_matrix[j, i] = 1
                else:
                    order_matrix[i, j] = 1
                    order_matrix[j, i] = -1
    return order_matrix

def infer_order_yaxis(inmodal):
    num = inmodal.shape[0]
    order_matrix = np.zeros((num, num), dtype=np.int)
    for i in range(num):
        for j in range(i + 1, num):
            if bordering(inmodal[i], inmodal[j]):
                center_i = [coord.mean() for coord in np.where(inmodal[i] == 1)] # y, x
                center_j = [coord.mean() for coord in np.where(inmodal[j] == 1)] # y, x
                if center_i[0] < center_j[0]: # i higher than j in y axis
                    order_matrix[i, j] = -1 # i occluded by j
                    order_matrix[j, i] = 1
                else:
                    order_matrix[i, j] = 1
                    order_matrix[j, i] = -1
    return order_matrix




def bordering(a, b):
    dilate_kernel = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=np.uint8)
    a_dilate = cv2.dilate(a.astype(np.uint8), dilate_kernel, iterations=1)
    return np.any((a_dilate == 1) & b)

def bbox_in(box1, box2):
    l1, u1, r1, b1 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    l2, u2, r2, b2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
    if l1 >= l2 and u1 >= u2 and r1 <= r2 and b1 <= b2:
        return True
    else:
        return False

def fullcovering(mask1, mask2, box1, box2):
    if not (mask1 == 0).all() and not (mask2 == 0).all():
        return 0
    if (mask1 == 0).all() and bbox_in(box1, box2): # 1 covered by 2
        return 1
    elif (mask2 == 0).all() and bbox_in(box2, box1):
        return 2
    else:
        return 0

def infer_gt_order(inmodal, amodal):
    #inmodal = inmodal.numpy()
    #amodal = amodal.numpy()
    num = inmodal.shape[0]
    gt_order_matrix = np.zeros((num, num), dtype=np.int)
    for i in range(num):
        for j in range(i + 1, num):
            if not bordering(inmodal[i], inmodal[j]):
                continue
            occ_ij = ((inmodal[i] == 1) & (amodal[j] == 1)).sum()
            occ_ji = ((inmodal[j] == 1) & (amodal[i] == 1)).sum()
            #assert not (occ_ij > 0 and occ_ji > 0) # assertion error, why?
            if occ_ij == 0 and occ_ji == 0: # bordering but not occluded
                continue
            gt_order_matrix[i, j] = 1 if occ_ij >= occ_ji else -1
            gt_order_matrix[j, i] = -gt_order_matrix[i, j]
    return gt_order_matrix

def eval_order(order_matrix, gt_order_matrix):
    inst_num = order_matrix.shape[0]
    allpair_true = ((order_matrix == gt_order_matrix).sum() - inst_num) / 2
    allpair = (inst_num  * inst_num - inst_num) / 2

    occpair_true = ((order_matrix == gt_order_matrix) & (gt_order_matrix != 0)).sum() / 2
    occpair = (gt_order_matrix != 0).sum() / 2

    err = np.where(order_matrix != gt_order_matrix)
    gt_err = gt_order_matrix[err]
    pred_err = order_matrix[err]
    show_err = np.concatenate([np.array(err).T + 1, gt_err[:,np.newaxis], pred_err[:,np.newaxis]], axis=1)
    return allpair_true, allpair, occpair_true, occpair, show_err

def get_neighbors(graph, idx):
    return np.where(graph[idx,:] != 0)[0]


def get_parents(graph, idx):
    return np.where(graph[idx,:] == -1)[0]


def get_neighbors_recur(graph, idx):
    is_neighbor = np.zeros((graph.shape[0],), dtype=np.bool)
    visited = np.zeros((graph.shape[0],), dtype=np.bool)
    queue = {idx}
    while len(queue) > 0:
        q = queue.pop()
        if visited[q]:
            continue # incase there exists cycles.
        visited[q] = True
        new_neighbor = np.where((graph[q, :] != 0) & (visited == 0))[0]
        is_neighbor[new_neighbor] = True
        queue.update(set(new_neighbor.tolist()))

    is_neighbor[idx] = False

    return np.where(is_neighbor)[0]


def get_ancestors(graph, idx):
    is_ancestor = np.zeros((graph.shape[0],), dtype=np.bool)
    visited = np.zeros((graph.shape[0],), dtype=np.bool)
    queue = {idx}
    while len(queue) > 0:
        q = queue.pop()
        if visited[q]:
            continue # incase there exists cycles.
        visited[q] = True
        new_ancestor = np.where(graph[q, :] == -1)[0]
        is_ancestor[new_ancestor] = True
        queue.update(set(new_ancestor.tolist()))
    is_ancestor[idx] = False
    return np.where(is_ancestor)[0]

def infer_instseg(model, image, category, bboxes, new_bboxes, input_size, th, rgb=None):
    num = bboxes.shape[0]
    seg_patches = []
    for i in range(num):
        rel_bbox = [bboxes[i,0] - new_bboxes[i,0],
                    bboxes[i,1] - new_bboxes[i,1], bboxes[i,2], bboxes[i,3]]
        bbox_mask = np.zeros((new_bboxes[i,3], new_bboxes[i,2]), dtype=np.uint8)
        bbox_mask[rel_bbox[1]:rel_bbox[1]+rel_bbox[3], rel_bbox[0]:rel_bbox[0]+rel_bbox[2]] = 1
        bbox_mask = cv2.resize(bbox_mask, (input_size, input_size),
            interpolation=cv2.INTER_NEAREST)
        bbox_mask_tensor = torch.from_numpy(
            bbox_mask.astype(np.float32) * category[i]).unsqueeze(0).unsqueeze(0).cuda()
        image_patch = cv2.resize(utils.crop_padding(image, new_bboxes[i], pad_value=(0,0,0)),
            (input_size, input_size), interpolation=cv2.INTER_CUBIC)
        image_tensor = torch.from_numpy(
            image_patch.transpose((2,0,1)).astype(np.float32)).unsqueeze(0).cuda() # 13HW
        with torch.no_grad():
            output = model.model(torch.cat([image_tensor, bbox_mask_tensor], dim=1)).detach()
        if output.shape[2] != image_tensor.shape[2]:
            output = nn.functional.interpolate(
                output, size=image_tensor.shape[2:4],
                mode="bilinear", align_corners=True) # 12HW
        output = nn.functional.softmax(output, dim=1) # 12HW
        if rgb is not None:
            prob = output[0,...].cpu().numpy() # 2HW
            rgb_patch = cv2.resize(utils.crop_padding(rgb, new_bboxes[i], pad_value=(0,0,0)),
                (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            prob_crf = np.array(utils.densecrf(prob, rgb_patch)).reshape(*prob.shape)
            pred = (prob_crf[1,:,:] > th).astype(np.uint8) # HW
        else:
            pred = (output[0,1,:,:] > th).cpu().numpy().astype(np.uint8) # HW
        seg_patches.append(pred)
    return seg_patches



def infer_amodal_aw_sdm(model, image_fn, inmodal, category, bboxes, use_rgb=True, th=0.5,
                     input_size=None, min_input_size=16, interp='nearest', debug_info=False, args=None):
    num = inmodal.shape[0]
    inmodal_patches = []
    amodal_patches = []

    import torch.nn as nn
    import os
    org_src_ft_dict = {}
    for layer_i in [0, 1, 2, 3]:
        feat_dir = 'pth' + str(layer_i)
        feat = torch.load(os.path.join(feat_dir, image_fn[:-4] + '.pt'))
        org_src_ft = feat.permute(1,2,0).numpy() # h x w x L
        org_src_ft_dict[layer_i] = org_src_ft
    org_h, org_w = inmodal[0].shape[0], inmodal[0].shape[1]

    for i in range(num):

        src_ft_dict = {}
        for layer_i in [0, 1, 2, 3]:
            org_src_ft = org_src_ft_dict[layer_i]
            src_ft_new_bbox = [
                int(bboxes[i][0] * org_src_ft.shape[1] / org_w),
                int(bboxes[i][1] * org_src_ft.shape[0] / org_h),
                int(bboxes[i][2] * org_src_ft.shape[1] / org_w),
                int(bboxes[i][3] * org_src_ft.shape[0] / org_h),
                ]
            src_ft = utils.crop_padding(org_src_ft, src_ft_new_bbox, pad_value=(0,)*org_src_ft.shape[-1])
            src_ft = torch.tensor(src_ft).permute(2,0,1).unsqueeze(0)
            src_ft = src_ft.to('cuda:0')
            if layer_i == 0:
                cur_upsample_sz = 24
            elif layer_i == 1:
                cur_upsample_sz = 48
            else:
                cur_upsample_sz = 96
            if src_ft.shape[-2] != 0 and src_ft.shape[-1] != 0:
                src_ft = nn.Upsample(size=(cur_upsample_sz, cur_upsample_sz), mode='bilinear')(src_ft).squeeze(0) # L x h x w
                src_ft = src_ft.permute(1,2,0).cpu().numpy() # h x w x L
            else:

                src_ft = torch.tensor(org_src_ft).permute(2,0,1).unsqueeze(0)
                src_ft = nn.Upsample(size=(org_h, org_w), mode='bilinear')(src_ft).squeeze(0) # L x h x w
                src_ft = src_ft.permute(1,2,0).cpu().numpy() # h x w x L
                src_ft = utils.crop_padding(src_ft, bboxes[i], pad_value=(0,)*src_ft.shape[-1]) # h x w x L
                src_ft = torch.tensor(src_ft).permute(2,0,1).unsqueeze(0)
                src_ft = nn.Upsample(size=(cur_upsample_sz, cur_upsample_sz), mode='bilinear')(src_ft).squeeze(0) # L x h x w
                src_ft = src_ft.permute(1,2,0).cpu().numpy() # h x w x L

            src_ft_dict[layer_i] = src_ft
            
        inmodal_patch = utils.crop_padding(inmodal[i], bboxes[i], pad_value=(0,))
        if input_size is not None:
            newsize = input_size
        elif min_input_size > bboxes[i,2]:
            newsize = min_input_size
        else:
            newsize = None
        if newsize is not None:
            inmodal_patch = resize_mask(inmodal_patch, newsize, interp)

        inmodal_patches.append(inmodal_patch)
        amodal_patches.append(net_forward_aw_sdm(
            model, src_ft_dict, inmodal_patch * category[i], None, use_rgb, th, args=args))
    if debug_info:
        return inmodal_patches, amodal_patches
    else:
        return amodal_patches



def patch_to_fullimage(patches, bboxes, height, width, interp):
    amodals = []
    for patch, bbox in zip(patches, bboxes):
        amodals.append(recover_mask(patch, bbox, height, width, interp))
    return np.array(amodals)
