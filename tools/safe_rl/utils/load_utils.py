#!/usr/bin/env python

import joblib
import os
import os.path as osp
import tensorflow as tf
from tools.safe_rl.utils.logx import restore_tf_graph

def loadmodel(session, saver, checkpoint_dir):
    session.run(tf.compat.v1.global_variables_initializer())
    tf.compat.v1.get_variable_scope().reuse_variables()
    ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

def load_policy(fpath, policy, config):

    # load the things!
    sess = config["sess"]
    saver = config["saver"]
    loadmodel(sess, saver, fpath)
    return policy