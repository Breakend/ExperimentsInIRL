# Note: This file is taken directly from https://raw.githubusercontent.com/cbfinn/gps/ssrl/python/gps/algorithm/cost/tf_cost_utils.py
#       and modified heavily for our purposes
import numpy as np
import tensorflow as tf


def assert_shape(tensor, shape):
    assert tensor.get_shape().is_compatible_with(shape), "Shape mismatch: %s vs %s" % (str(tensor.get_shape()), shape)

def logsumexp(x, reduction_indices=None):
    """ Compute numerically stable logsumexp """
    max_val = tf.reduce_max(x)
    exp = tf.exp(x-max_val)
    _partition = tf.reduce_sum(exp, reduction_indices=reduction_indices)
    _log = tf.log(_partition)+max_val
    return _log

def safe_get(name, *args, **kwargs):
    """ Same as tf.get_variable, except flips on reuse_variables automatically """
    try:
        return tf.get_variable(name, *args, **kwargs)
    except ValueError:
        tf.get_variable_scope().reuse_variables()
        return tf.get_variable(name, *args, **kwargs)

def init_weights(shape, name=None):
    weights = np.random.normal(scale=0.01, size=shape).astype('f')
    return safe_get(name, list(shape), initializer=tf.constant_initializer(weights))

def init_bias(shape, name=None):
    return safe_get(name, initializer=tf.zeros(shape, dtype='float'))

def find_variable(name):
    """ Find a trainable variable in a graph by its name (not including scope)

    Example:
    >>> find_variable('Wconv1')
    <tensorflow tensor>
    """
    varnames = tf.trainable_variables()
    matches = [varname for varname in varnames if varname.name.endswith(name+':0')]

    if len(matches)>1:
        raise ValueError('More than one variable with name %s. %s' % (name, [match.name for match in matches]))
    if len(matches) == 0:
        raise ValueError('No variables found with name %s. List: %s' % (name, [var.name for var in varnames]))
    return matches[0]

def jacobian(y, x):
    """Compute derivative of y (vector) w.r.t. x (another vector)"""
    dY = y.get_shape()[0].value
    dX = x.get_shape()[0].value

    deriv_list = []
    for idx_y in range(dY):
        grad = tf.gradients(y[idx_y], x)[0]
        deriv_list.append(grad)
    jac = tf.pack(deriv_list)
    assert_shape(jac, [dY, dX])
    return jac

def compute_feats(net_input, num_hidden=1, dim_hidden=42):
    len_shape = len(net_input.get_shape())
    if  len_shape == 3:
        batch_size, T, dinput = net_input.get_shape()
    elif len_shape == 2:
        T, dinput = net_input.get_shape()

    # Reshape into 2D matrix for matmuls
    net_input = tf.reshape(net_input, [-1, dinput.value])
    with tf.variable_scope('cost_forward'):
        layer = net_input
        for i in range(num_hidden-1):
            with tf.variable_scope('layer_%d' % i):
                # W = safe_get('W', shape=(dim_hidden, layer.get_shape()[1].value))
                # b = safe_get('b', shape=(dim_hidden))
                W = init_weights((dim_hidden, layer.get_shape()[1].value), name='W')
                b = init_bias((dim_hidden), name='b')
                layer = tf.nn.relu(tf.matmul(layer, W, transpose_b=True, name='mul_layer'+str(i)) + b)

        Wfeat = init_weights((dim_hidden, layer.get_shape()[1].value), name='Wfeat')
        bfeat = init_bias((dim_hidden), name='bfeat')
        feat = tf.matmul(layer, Wfeat, transpose_b=True, name='mul_feat')+bfeat

    if len_shape == 3:
        feat = tf.reshape(feat, [batch_size.value, T.value, dim_hidden])
    else:
        feat = tf.reshape(feat, [-1, dim_hidden])

    return feat

def nn_forward(net_input, num_hidden=1, dim_hidden=42, learn_wu=False):
    # Reshape into 2D matrix for matmuls
    # u_input = tf.reshape(u_input, [-1, 1])

    feat = compute_feats(net_input, num_hidden=num_hidden, dim_hidden=dim_hidden)
    feat = tf.reshape(feat, [-1, dim_hidden])

    with tf.variable_scope('cost_forward'):
        # A = safe_get('Acost', shape=(dim_hidden, dim_hidden))
        # b = safe_get('bcost', shape=(dim_hidden))
        A = init_weights((dim_hidden, dim_hidden), name='Acost')
        b = init_bias((dim_hidden), name='bcost')
        Ax = tf.matmul(feat, A, transpose_b=True)+b
        AxAx = Ax*Ax

        # Calculate torque penalty
        # u_penalty = safe_get('wu', initializer=tf.constant(1.0), trainable=learn_wu)
        # assert_shape(u_penalty, [])
        # u_cost = u_input*u_penalty

    # Reshape result back into batches
    input_shape = net_input.get_shape()
    if len(input_shape) == 3:
        batch_size, T, dinput = input_shape
        batch_size, T = batch_size.value, T.value
        AxAx = tf.reshape(AxAx, [batch_size, T, dim_hidden])
        # u_cost = tf.reshape(u_cost, [batch_size, T, 1])
    elif len(input_shape) == 2:
        AxAx = tf.reshape(AxAx, [-1, dim_hidden])
        # u_cost = tf.reshape(u_cost, [-1, 1])
    all_costs_preu = tf.reduce_sum(AxAx, reduction_indices=[-1], keep_dims=True)
    all_costs = all_costs_preu #+ u_cost
    return all_costs_preu, all_costs

def conv2d(img, w, b, strides=[1, 1, 1, 1]):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=strides, padding='SAME'), b))

def icml_loss(expert_costs, sample_costs):
    num_demos, T, _ = expert_costs.get_shape()
    num_samples, T, _ = sample_costs.get_shape()

    # Sum over time and compute max value for safe logsum.
    demo_reduced = 0.5*tf.reduce_sum(expert_costs, reduction_indices=[1,2])
    dc = demo_reduced # + tf.reduce_sum(d_log_iw, reduction_indices=[1])
    assert_shape(dc, [num_demos])

    sc = 0.5*tf.reduce_sum(sample_costs, reduction_indices=[1,2])# + tf.reduce_sum(s_log_iw, reduction_indices=[1])
    assert_shape(sc, [num_samples])

    dc_sc = tf.concat(axis=0, values=[-dc, -sc])

    loss = tf.reduce_mean(demo_reduced)

    # Concatenate demos and samples to approximate partition function
    partition_samples = tf.concat(axis=0, values=[expert_costs, sample_costs])
    #partition_iw = tf.concat(0, [d_log_iw, s_log_iw])
    partition = 0.5*tf.reduce_sum(partition_samples, reduction_indices=[1,2]) #+tf.reduce_sum(partition_iw, reduction_indices=[1])
    assert_shape(partition, [num_samples+num_demos])
    loss += logsumexp(-partition, reduction_indices=[0])

    assert_shape(loss, [])
    return loss

def l2_mono_loss(slope):
    offset = 1.0
    bottom_data = slope

    _temp = tf.nn.relu(bottom_data+offset)
    loss = tf.nn.l2_loss(_temp)
    return loss
