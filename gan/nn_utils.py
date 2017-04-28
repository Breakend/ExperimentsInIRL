import tensorflow as tf


def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

def logit_bernoulli_entropy(logits_B):
    ent_B = (1.-tf.sigmoid(logits_B))*logits_B - logsigmoid(logits_B)
    return ent_B
