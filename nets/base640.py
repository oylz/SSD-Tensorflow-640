import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import ssd_common

slim = tf.contrib.slim


# only used in flow function:ssd_anchors_all_layers(), final for outapp to detect
def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = 1 #xxxxxxxx1 #xxxxxxxxxxxxxxxxxxxxxxxxxxxxx len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    #if len(sizes) > 1:
    if num_anchors > 1: # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    #else: # xxxxadd elsexxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #    h[1] = img_shape[0]
    #    w[1] = img_shape[1]
    #    di += 1

    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w

# only used in SSDNet.anchors(),final for outapp to detect
def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

#===========================================================================

# used in SSDNet.net() & SSDNet.update_feature_shapes()
def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]
        # Problem: undetermined shape...
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes

# only used in ssd_multibox_layer
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

# only used in ssd_net
def ssd_multibox_layer(addn, inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    num_anchors = sizes  #len(sizes) + len(ratios)
    print("ssd_multibox_layer0")
    print(normalization)
    print("ssd_multibox_layer1")
    print(sizes)
    print("ssd_multibox_layer2")
    print(ratios)
    print("ssd_multibox_layer3")
    # Location.
    num_loc_pred = 4 #xxxxxxxxxxxxxxxxxxxx sizes  #num_anchors * 4
    print(num_loc_pred)
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                           scope='conv_loc')
    print(loc_pred)
    print("ssd_multibox_layer4")
    loc_pred = custom_layers.channel_to_last(loc_pred)
    print(loc_pred)
    print("ssd_multibox_layer5")
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    print(loc_pred)
    print("ssd_multibox_layer6")
    # Class prediction.
    num_cls_pred = (num_anchors+addn)*2 #xxxxxxxxxxxxxxxxxxxxxxxx num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls')
    print(num_cls_pred)
    print(cls_pred)
    print("ssd_multibox_layer7")
    cls_pred = custom_layers.channel_to_last(cls_pred)
    print(cls_pred)
    print("ssd_multibox_layer8")
    tt = tensor_shape(cls_pred, 4)
    print(tt)
    #cls_pred = tf.reshape(cls_pred,
    #                      tt[:-1]+[num_anchors, num_classes])
    cls_pred = tf.reshape(cls_pred,
                          tt[:-1]+[num_anchors+addn, num_classes])
    print(cls_pred)
    print("ssd_multibox_layer9")
    return cls_pred, loc_pred

# only used in SSDNet.net(), and extend for nets_factory
def ssd_net(inputs,
            num_classes,
            feat_layers,
            anchor_sizes,
            anchor_ratios,
            normalizations,
            is_training,
            dropout_keep_prob,
            prediction_fn,
            reuse,
            scope):
    """SSD net definition.
    """
    # if data_format == 'NCHW':
    #     inputs = tf.transpose(inputs, perm=(0, 3, 1, 2))

    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'ssd_640_vgg', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        print("block1 begin")
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='r2_crcr1')
        end_points['block1'] = net
        print("block1 end")

        print("block2 begin")
        net = slim.max_pool2d(net, [2, 2], scope='bbpool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='r2_crcr2')
        end_points['block2'] = net
        print("block2 end")


        print("block3 begin")
        net = slim.max_pool2d(net, [2, 2], scope='ddpool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='r3_crcr3')
        end_points['block3'] = net
        print("block3 end")


        print("block4 begin")
        net = slim.max_pool2d(net, [2, 2], scope='ffpool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='r3_crcr4')
        end_points['block4'] = net
        print("block4 end")


        print("block5 begin")
        net = slim.max_pool2d(net, [2, 2], scope='hhpool4')
        # rate as `[dilation]`/`pad` in prototxt?, if is `[dilation]` then set rate=1
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], rate=1, scope='r3_crcr5')
        end_points['block5'] = net
        print("block5 end")


        print("block6 begin")
        # pool5: kernel_size: 3->2, stride: 1->2, +pad:1,, where to put `pad:1`?
        net = slim.max_pool2d(net, [2, 2], stride=2, scope='jjpool5')
        net = slim.conv2d(net, 1024, [3, 3], rate=1, scope='kkfc6')
        end_points['block6'] = net
        print("block6 end")


        print("block7 begin")
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
        net = slim.conv2d(net, 1024, [1, 1], scope='llfc7')
        end_points['block7'] = net
        print("block7 end")


        print("block8 begin")
        # conv61->conv62
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
        end_point = 'block8'
        with tf.variable_scope(end_point):
            # paper: 1x1x128
            net = slim.conv2d(net, 256, [1, 1], scope='mmconv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            # paper: 3x3x512-s2
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='nnconv3x3', padding='VALID')
        end_points[end_point] = net


        end_point = 'block9'
        # conv71->conv72
        with tf.variable_scope(end_point):
            # paper: 1x1x128
            net = slim.conv2d(net, 128, [1, 1], scope='ooconv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            # paper: 3x3x256-s2
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='ppconv3x3', padding='VALID')
        end_points[end_point] = net
        print("block9 end")


        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        addn = 1
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                print("process----" + layer + '_box')
                p, l = ssd_multibox_layer(addn, end_points[layer],
                                          num_classes,
                                          1,  #anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
                addn = 0

            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)
        print("final end")
        return predictions, localisations, logits, end_points
ssd_net.default_image_size = 640




# only used in SSDNet.losses()
def ssd_losses(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               device='/cpu:0',
               scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        lshape = tfe.get_shape(logits[0], 5)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        dtype = logits.dtype

        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask)

        # Hard negative mining...
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)
        nmask = tf.logical_and(tf.logical_not(pmask),
                               gscores > -0.5)
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=gclasses)
            loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)
            loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localisations - glocalisations)
            loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
            tf.losses.add_loss(loss)

#==========================================================
#used in SSDNet, and extend for nets_factory
def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc


#used in SSDNet, and extend for nets_factory
def ssd_arg_scope_caffe(caffe_scope):
    """Caffe scope definition.

    Args:
      caffe_scope: Caffe scope object with loaded weights.

    Returns:
      An arg_scope.
    """
    # Default network arg scope.
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=caffe_scope.conv_weights_init(),
                        biases_initializer=caffe_scope.conv_biases_init()):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([custom_layers.l2_normalization],
                                scale_initializer=caffe_scope.l2_norm_scale_init()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    return sc


