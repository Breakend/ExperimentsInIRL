import tensorflow as tf

TINY = 1.0e-8

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

def logit_bernoulli_entropy(logits_B):
    ent_B = (1.-tf.sigmoid(logits_B))*logits_B - logsigmoid(logits_B)
    return ent_B

def log10(x):
    numerator = tf.log(tf.clip_by_value(x, 1.0e-8, 99999))
    #TODO: Use a sensical value for upper bound    ^
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars]) # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
