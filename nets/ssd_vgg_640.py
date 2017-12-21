import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe

from nets import  base640


SSDParams = namedtuple('SSDParameters', ['img_shape', 
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class SSDNet(object):
    default_params = SSDParams(
        img_shape=(640, 640),
        num_classes=2,
        no_annotation_label=2,
        feat_layers=['block3', 'block4', 'block5', 'block7', 'block8', 'block9'],
        feat_shapes=[(160, 160), (80, 80), (40, 40), (20, 20), (10, 10), (5, 5)],
        anchor_size_bounds=[0.15, 0.90],
        anchor_sizes=[(16.),
                      (32.),
                      (64.),
                      (128.),
                      (256.),
                      (512.)],
        anchor_ratios=[[],
                       [],
                       [],
                       [],
                       [],
                       []],

        anchor_steps=[4, 8, 16, 32, 64, 128],
        anchor_offset=0.5,
        normalizations=[10, 8, 5, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.slim.softmax,
            reuse=None,
            scope='ssd_640_vgg'):
        r = base640.ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        print("update_feat_shapes:", update_feat_shapes)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = base640.ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        return base640.ssd_arg_scope(weight_decay, data_format=data_format)


    def arg_scope_caffe(self, caffe_scope):
        return base640.ssd_arg_scope_caffe(caffe_scope)


    # for outapp to detect
    def anchors(self, img_shape, dtype=np.float32):
        return base640.ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)





