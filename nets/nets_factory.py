
import functools
import tensorflow as tf

# from nets import inception
# from nets import overfeat
# from nets import resnet_v1
# from nets import resnet_v2
from nets import vgg
# from nets import xception

from nets import ssd_vgg_640
from nets import  base640


slim = tf.contrib.slim

networks_map = {'vgg_a': vgg.vgg_a,
                'vgg_16': vgg.vgg_16,
                'vgg_19': vgg.vgg_19,
                'ssd_640_vgg': base640.ssd_net,
                'ssd_640_vgg_caffe': base640.ssd_net,
                }

arg_scopes_map = {'vgg_a': vgg.vgg_arg_scope,
                  'vgg_16': vgg.vgg_arg_scope,
                  'vgg_19': vgg.vgg_arg_scope,
                  'ssd_640_vgg': base640.ssd_arg_scope,
                  'ssd_640_vgg_caffe': base640.ssd_arg_scope_caffe,
                  }

networks_obj = {'ssd_640_vgg': ssd_vgg_640.SSDNet,
                }


def get_network(name):
    """Get a network object from a name.
    """
    # params = networks_obj[name].default_params if params is None else params
    return networks_obj[name]


def get_network_fn(name, num_classes, is_training=False, **kwargs):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification.
      is_training: `True` if the model is being used for training and `False`
        otherwise.
      weight_decay: The l2 coefficient for the model weights.
    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature: logits, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    arg_scope = arg_scopes_map[name](**kwargs)
    func = networks_map[name]
    @functools.wraps(func)
    def network_fn(images, **kwargs):
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training, **kwargs)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
