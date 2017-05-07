from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.misc import tensor_utils
import numpy as np
import tensorflow as tf


class LinearFeatureBaseline(Baseline):
    def __init__(self, env_spec, reg_coeff=1e-5):
        self._coeffs = None
        self._reg_coeff = reg_coeff
        self.feature_mat = tensor_utils.new_tensor(
            'feature_mat',
            ndim=2,
            dtype=tf.float32,
        )
        self.returns = tensor_utils.new_tensor(
            'returns',
            ndim=2,
            dtype=tf.float32,
        )
        # import pdb; pdb.set_trace()
        ident = tf.identity(self.feature_mat)
        self.train_ops = tf.matrix_solve_ls(tf.square(self.feature_mat) + self._reg_coeff * ident, self.returns, fast=False)
        self.sess = tf.Session()


    @overrides
    def get_param_values(self, **tags):
        return self._coeffs

    @overrides
    def set_param_values(self, val, **tags):
        self._coeffs = val

    def _features(self, path):
        o = np.clip(path["observations"], -10, 10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

    @overrides
    def fit(self, paths, sess=None):
        # created_session = True if (sess is None) else False
        # if sess is None:
        #     sess = tf.Session()
        #     sess.__enter__()
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = self.sess.run([self.train_ops], feed_dict={self.feature_mat : featmat, self.returns : returns.reshape(-1, 1)})
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10
        # if created_session:
        #     sess.close()


    @overrides
    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        return self._features(path).dot(self._coeffs)
