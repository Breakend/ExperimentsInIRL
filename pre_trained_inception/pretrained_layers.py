import numpy as np
import os
import tensorflow as tf

from . import inception_v4
from . import inception_utils
from . import inception_preprocessing

slim = tf.contrib.slim
image_size = inception_v4.inception_v4.default_image_size

def pre_trained_net(input_dims, checkpoint_dir):
    inputs = tf.placeholder('float', input_dims, name='nn_input')

    processed_image = inception_preprocessing.preprocess_image(inputs,
                                                     image_size,
                                                     image_size,
                                                     is_training=False)
    processed_image  = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(inception_utils.inception_arg_scope()):
        last_layer = inception_v4.inception_v4(processed_image,
                            #    num_classes=1001, #number of features we want ?
                               is_training=False)
        # last_layer.reshape(1, -1)
        # Get the current scope, if it's not the top level,
        # need to append / to get the next InceptionV4 layer vars
        sc = tf.get_variable_scope().name
        if sc:
            sc += "/"
        variables_in_graph = slim.get_model_variables('%sInceptionV4' % sc)
        checkpoint_map = {}
        # TODO: hack hack hack
        # import pdb; pdb.set_trace()
        for v in variables_in_graph:
            checkpoint_map[v.name.strip(sc).strip(":0").strip(":")] = v

        init_fn = slim.assign_from_checkpoint_fn(
                    checkpoint_dir,
                    var_list=checkpoint_map)
    return init_fn, inputs, last_layer
