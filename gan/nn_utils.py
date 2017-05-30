import tensorflow as tf
import time
import rllab.misc.logger as logger

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

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def custom_train(algo, sess=None):
    created_session = True if (sess is None) else False
    if sess is None:
        sess = tf.Session()
        sess.__enter__()

    rollout_cache = []
    initialize_uninitialized(sess)
    algo.start_worker()
    start_time = time.time()
    for itr in range(algo.start_itr, algo.n_itr):
        itr_start_time = time.time()
        with logger.prefix('itr #%d | ' % itr):
            logger.log("Obtaining samples...")
            paths = algo.obtain_samples(itr)
            logger.log("Processing samples...")
            samples_data = algo.process_samples(itr, paths)
            logger.log("Logging diagnostics...")
            algo.log_diagnostics(paths)
            logger.log("Optimizing policy...")
            algo.optimize_policy(itr, samples_data)
            logger.log("Saving snapshot...")
            params = algo.get_itr_snapshot(itr, samples_data)  # , **kwargs)
            if algo.store_paths:
                params["paths"] = samples_data["paths"]
            logger.save_itr_params(itr, params)
            logger.log("Saved")
            logger.record_tabular('Time', time.time() - start_time)
            logger.record_tabular('ItrTime', time.time() - itr_start_time)
            logger.dump_tabular(with_prefix=False)

    algo.shutdown_worker()
    if created_session:
        sess.close()


def lrelu(x, alpha=.01, max_value=None):
    '''LeakyReLU.

    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=ft.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x
