import os
import sys
import random
import math
import numpy as np
import skimage.io
import cv2 as cv
import coco
import utils
import train_2
import model as modellib
import visualize
from config import Config

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



ROOT_DIR = os.path.abspath("../")
MODEL_DIR = os.path.join(ROOT_DIR, "mrcnn/logs")
MODEL_WEIGHT_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_shapes_0040.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "ce_shi_ji")



class somethingConfig(Config):
    NAME = "something"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1


class InferenceConfig(somethingConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(MODEL_WEIGHT_PATH, by_name=True)
class_names = ['BG', 'fingerprint1']
file_names = next(os.walk(IMAGE_DIR))[2]


for x in range(len(file_names)):
    image_ = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[x]))
    results = model.detect([image_], verbose=1)
    #dataset = train_2.DrugDataset()
    r = results[0]
    # image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
    #     dataset, config, image_, use_mini_mask=False)
    #visualize.display_instances(image, bbox, mask, class_ids, class_names)


    visualize.display_instances(image_, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

    # 画出precision-recall的曲线
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                         r['rois'], r['class_ids'], r['scores'], r['masks'])
    visualize.plot_precision_recall(AP, precisions, recalls)

    # 显示ground truth和预测的网格
    visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
                            overlaps, class_names)

    # 生成RPN trainig targets
    # target_rpn_match=1是positive anchors, -1是negative anchors
    # 0是neutral anchors.
    target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
        image.shape, model.anchors, gt_class_id, gt_bbox, model.config)
    log("target_rpn_match", target_rpn_match)
    log("target_rpn_bbox", target_rpn_bbox)

    positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
    negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
    neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
    positive_anchors = model.anchors[positive_anchor_ix]
    negative_anchors = model.anchors[negative_anchor_ix]
    neutral_anchors = model.anchors[neutral_anchor_ix]
    log("positive_anchors", positive_anchors)
    log("negative_anchors", negative_anchors)
    log("neutral anchors", neutral_anchors)

    # 将refinement deltas应用于positive anchors
    refined_anchors = utils.apply_box_deltas(
        positive_anchors,
        target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
    log("refined_anchors", refined_anchors, )

    # 显示refinement (点)之前的positive anchors和refinement (线)之后的positive anchors.
    visualize.draw_boxes(image, boxes=positive_anchors, refined_boxes=refined_anchors, ax=get_ax())
