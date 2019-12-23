import os
import sys
import random
import math
import numpy as np
import skimage.io
import coco
import utils
from samples.shapes import shapes
import model as modellib
import visualize
from config import Config

from mrcnn.model import log

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



ROOT_DIR = os.path.abspath("../")
MODEL_DIR = os.path.join(ROOT_DIR, "mrcnn/logs")
MODEL_WEIGHT_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_shapes_0040.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "Luping_fingerprint")



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

dataset = shapes.ShapesDataset()
for x in range(len(file_names)):
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[x]))
    # image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    #     modellib.load_image_gt(dataset, config,
    #                            image, use_mini_mask=False)

    # image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, config,
    #                                                                                    image_id=image_id,
    #                                                                                    use_mini_mask=False)



    results = model.detect([image], verbose=1)

    r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                             class_names, r['scores'])


    # visualize.display_images(np.transpose(gt_mask, [2, 0, 1]), cmap="Blues")

    # 获取mask分支的预测结果
    mrcnn = model.run_graph([image], [
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
    ])

    # 获取检测结果的class IDs.修剪zero padding.
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    det_count = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:det_count]

    print("{} detections: {}".format(
        det_count, np.array(class_names)[det_class_ids]))

    # Masks
    det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
    det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c]
                                  for i, c in enumerate(det_class_ids)])


    det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
                          for i, m in enumerate(det_mask_specific)])
    log("det_mask_specific", det_mask_specific)
    log("det_masks", det_masks)


    #visualize.display_images(det_mask_specific[:4] * 255, cmap="Blues", interpolation="none")

    visualize.display_images(det_masks[:4] * 255, cmap="Blues", interpolation="none")
