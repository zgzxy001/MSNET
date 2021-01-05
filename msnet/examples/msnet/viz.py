# -*- coding: utf-8 -*-
# File: viz.py

import numpy as np

from tensorpack.utils import viz
from tensorpack.utils.palette import PALETTE_RGB

from config import config as cfg
from utils.np_box_ops import area as np_area
from utils.np_box_ops import iou as np_iou
from common import polygons_to_mask


def draw_annotation(img, boxes, klass, polygons=None, is_crowd=None):
    """Will not modify img"""
    labels = []
    assert len(boxes) == len(klass)
    if is_crowd is not None:
        assert len(boxes) == len(is_crowd)
        for cls, crd in zip(klass, is_crowd):
            clsname = cfg.DATA.CLASS_NAMES[cls]
            if crd == 1:
                clsname += ';Crowd'
            labels.append(clsname)
    else:
        for cls in klass:
            labels.append(cfg.DATA.CLASS_NAMES[cls])
    img = viz.draw_boxes(img, boxes, labels)

    if polygons is not None:
        for p in polygons:
            mask = polygons_to_mask(p, img.shape[0], img.shape[1])
            img = draw_mask(img, mask)
    return img


def draw_proposal_recall(img, proposals, proposal_scores, gt_boxes):
    """
    Draw top3 proposals for each gt.
    Args:
        proposals: NPx4
        proposal_scores: NP
        gt_boxes: NG
    """
    box_ious = np_iou(gt_boxes, proposals)    # ng x np
    box_ious_argsort = np.argsort(-box_ious, axis=1)
    good_proposals_ind = box_ious_argsort[:, :3]   # for each gt, find 3 best proposals
    good_proposals_ind = np.unique(good_proposals_ind.ravel())

    proposals = proposals[good_proposals_ind, :]
    tags = list(map(str, proposal_scores[good_proposals_ind]))
    img = viz.draw_boxes(img, proposals, tags)
    return img, good_proposals_ind


def draw_predictions(img, boxes, scores):
    """
    Args:
        boxes: kx4
        scores: kxC
    """
    if len(boxes) == 0:
        return img
    labels = scores.argmax(axis=1)
    scores = scores.max(axis=1)
    tags = ["{},{:.2f}".format(cfg.DATA.CLASS_NAMES[lb], score) for lb, score in zip(labels, scores)]
    return viz.draw_boxes(img, boxes, tags)


def cal_mask_iou(component1, component2):
    component1 = np.array(component1, dtype=bool)
    component2 = np.array(component2, dtype=bool)
    overlap = component1 * component2  # Logical AND
    union = component1 + component2  # Logical OR

    IOU = overlap.sum() / float(union.sum())  # Treats "True" as 1,
    return IOU


# https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def class_nms(results, sorted_inds, iou_th = 0.1):
    lst = []
    for rst_id_1 in sorted_inds:
        for rst_id_2 in sorted_inds:
            if rst_id_1 != rst_id_2:
                rst_1 = results[rst_id_1]
                rst_2 = results[rst_id_2]
                iou = cal_mask_iou(rst_1.mask, rst_2.mask)
                if iou > iou_th:
                    if rst_1.score > rst_2.score:
                        lst.append(rst_id_2)
                    else:
                        lst.append(rst_id_1)
    return list(set(lst))


def cal_box_iou(box1, box2):
    box1 = {'x1': box1[0], 'y1': box1[1], 'x2': box1[2], 'y2':box1[3]}
    box2 = {'x1': box2[0], 'y1': box2[1], 'x2': box2[2], 'y2': box2[3]}
    iou = get_iou(box1, box2)
    return iou


def box_class_nms(results, sorted_inds, iou_th = 0.1):
    lst = []
    for rst_id_1 in sorted_inds:
        for rst_id_2 in sorted_inds:
            if rst_id_1 != rst_id_2:
                rst_1 = results[rst_id_1]
                rst_2 = results[rst_id_2]
                iou = cal_box_iou(rst_1.box, rst_2.box)
                if iou > iou_th:
                    if rst_1.score > rst_2.score:
                        lst.append(rst_id_2)
                    else:
                        lst.append(rst_id_1)
    return list(set(lst))



    # ttl_len = len(results)
    # for i in range(ttl_len):

def draw_final_outputs(img, results):
    """
    Args:
        results: [DetectionResult]
    """
    # new_results = []
    # for r in results:
    #     if r.score <=0.49:
    #         new_results.append(r)
    # results = new_results
    if len(results) == 0:
        return img

    # Display in largest to smallest order to reduce occlusion
    boxes = np.asarray([r.box for r in results])
    areas = np_area(boxes)
    sorted_inds = np.argsort(-areas)

    ret = img
    tags = []

    new_boxes = []
    # rm_lst = class_nms(results, sorted_inds)
    rm_lst = box_class_nms(results, sorted_inds)
    print("rm_lst = ", rm_lst)

    for result_id in sorted_inds:
        if result_id in rm_lst:
            continue
        r = results[result_id]

        # print("r = ", r)
        if r.mask is not None:
            level = str(r.class_id).split(" ")[0]
            if "1" in level:
                # color = (0, 255, 0)
                # color = [0.000, 255.000, 0.000]
                color_id = 23
                # color_id = 9
            elif "2" in level:
                color_id = 22
                # color_id = 9
                # color = [0.000, 255.000, 255.000]
                # color = (0, 255, 255)
            elif "3" in level:
                color_id = 9
                # color = [0.000, 0.000, 255.000]
                # color = (0, 0, 255)
            else:
                color = [0.000, 255.000, 0.000]
                # color = (0, 255, 0)
                print("error level!")
            ret = draw_mask(ret, r.mask, color=None, color_id=color_id)

    for result_id in sorted_inds:
        if result_id in rm_lst:
            continue
        r = results[result_id]
        new_boxes.append(r.box)
        tags.append(
            "{}, {:.2f}".format(cfg.DATA.CLASS_NAMES[r.class_id], r.score))
    # for r in results:
    #     tags.append(
    #         "{}, {:.2f}".format(cfg.DATA.CLASS_NAMES[r.class_id], r.score))
    ret = viz.draw_boxes(ret, new_boxes, tags)
    return ret


def draw_final_outputs_blackwhite(img, results):
    """
    Args:
        results: [DetectionResult]
    """
    img_bw = img.mean(axis=2)
    img_bw = np.stack([img_bw] * 3, axis=2)

    if len(results) == 0:
        return img_bw

    boxes = np.asarray([r.box for r in results])

    all_masks = [r.mask for r in results]
    if all_masks[0] is not None:
        m = all_masks[0] > 0
        for m2 in all_masks[1:]:
            m = m | (m2 > 0)
        img_bw[m] = img[m]

    tags = ["{},{:.2f}".format(cfg.DATA.CLASS_NAMES[r.class_id], r.score) for r in results]
    ret = viz.draw_boxes(img_bw, boxes, tags)
    return ret


def draw_mask(im, mask, alpha=0.5, color=None,color_id=None):
    """
    Overlay a mask on top of the image.
    Args:
        im: a 3-channel uint8 image in BGR
        mask: a binary 1-channel image of the same size
        color: if None, will choose automatically
    """
    # if color is None:
    #     color = PALETTE_RGB[np.random.choice(len(PALETTE_RGB))][::-1]
    # if color is None:
    color = PALETTE_RGB[color_id][::-1]
    color = np.asarray(color, dtype=np.float32)
    im = np.where(np.repeat((mask > 0)[:, :, None], 3, axis=2),
                  im * (1 - alpha) + color * alpha, im)
    im = im.astype('uint8')
    return im
