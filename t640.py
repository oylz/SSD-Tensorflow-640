

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append('../')

from nets import ssd_vgg_640, np_methods
from preprocessing import ssd_vgg_preprocessing
#from notebooks import visualization
import cv2
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm
from nets import nets_factory


# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (640, 640)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_640.SSDNet()
#ssd_class = nets_factory.get_network("ssd_640_vgg")
#ssd_params = ssd_class.default_params._replace(num_classes=2)
#ssd_net = ssd_class(ssd_params)
#ssd_shape = ssd_net.params.img_shape
#shape = (1, ssd_shape[0], ssd_shape[1], 3)
#img_input = tf.placeholder(shape=shape, dtype=tf.float32)

print("+++++++++++++++++++++++++++++++++++++")
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
    #predictions, localisations, _, _ = ssd_net.net(img_input, is_training=False)

print("======================mmmmmmmmmmmmmmmmmmmmm==========")
# Restore SSD model.
ckpt_filename = './tmp640/SFD.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)





# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(640, 640)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run(
                            [image_4d, predictions, localisations, bbox_img],
                            feed_dict={img_input: img})
    
    # TreateBoxes
    rclasses, rscores, rbboxes = np_methods.TreateBoxes(
                       rpredictions, 
                       rlocalisations, 
                       ssd_anchors,
                       select_threshold=select_threshold, 
                       img_shape=net_shape, 
                       num_classes=2, 
                       decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, 
                                            rscores, 
                                            rbboxes, 
                                            top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, 
                                            rscores, 
                                            rbboxes, 
                                            nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes



def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()




# Test on some demo image and visualize output.
path = './demo/'
image_names = sorted(os.listdir(path))

#pp =  path + image_names[-5]
pp = "/home/xyz/code1/xyz/img1/000117.jpg"

print("=======[[[[[[" + pp + "]]]]]]========")
img = mpimg.imread(pp)

rclasses, rscores, rbboxes =  process_image(img)

# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
plt_bboxes(img, rclasses, rscores, rbboxes)







