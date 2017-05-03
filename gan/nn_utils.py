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
