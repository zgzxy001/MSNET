from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import pickle
import numpy as np

user_flags = []



def reshape_features_chw(feature_0, feature_1, feature_2, feature_3):
    # Convert each feature map from 1xHxWxC to 1xCxHxW
    feature_0 = feature_0.swapaxes(2, 3).swapaxes(1, 2)
    feature_1 = feature_1.swapaxes(2, 3).swapaxes(1, 2)
    feature_2 = feature_2.swapaxes(2, 3).swapaxes(1, 2)
    feature_3 = feature_3.swapaxes(2, 3).swapaxes(1, 2)
    assert np.shape(feature_0)[1] == 1024
    return [feature_0, feature_1, feature_2, feature_3]


def reshape_features_hwc(feature):
    # Convert each feature map from CxHxW to HxWxC
    feature_0 = tf.transpose(feature, [1, 2, 0])
    return feature_0

def DEFINE_string(name, default_value, doc_string):
  tf.compat.v1.flags.DEFINE_string(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_integer(name, default_value, doc_string):
  tf.compat.v1.flags.DEFINE_integer(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_float(name, default_value, doc_string):
  tf.compat.v1.flags.DEFINE_float(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_boolean(name, default_value, doc_string):
  tf.compat.v1.flags.DEFINE_boolean(name, default_value, doc_string)
  global user_flags
  user_flags.append(name)


def print_user_flags(line_limit=80):
  print("-" * 80)

  global user_flags
  FLAGS = tf.compat.v1.flags.FLAGS

  for flag_name in sorted(user_flags):
    value = "{}".format(getattr(FLAGS, flag_name))
    log_string = flag_name
    log_string += "." * (line_limit - len(flag_name) - len(value))
    log_string += value
    print(log_string)


def load_val_lst(max_val):
    return list(range(1, max_val+1))



