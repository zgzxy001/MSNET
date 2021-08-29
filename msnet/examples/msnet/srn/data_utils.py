from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os

import numpy as np
import tensorflow as tf


def get_trip(pickle_file):
    with open(pickle_file, "rb") as f:
        obj = pickle.load(f)
    new_lst = []
    for video_dict in obj:
        video_name = list(video_dict.keys())[0]
        trip_lst = video_dict[video_name]
        for trip in trip_lst:
            new_lst.append((video_name+"_"+str(trip[0]), video_name+"_"+str(trip[1]), video_name+"_"+str(trip[2])))
    return new_lst


def load_feature(main_path, frame_name):
    frame_path = main_path + str(frame_name) + ".pickle"
    with open(frame_path, "rb") as rf:
        obj = pickle.load(rf)
    curr_dict = obj[list(obj.keys())[0]]
    feature_0, feature_1, feature_2, feature_3 = np.asarray(curr_dict[0]), np.asarray(curr_dict[1]), np.asarray(curr_dict[2]), np.asarray(curr_dict[3])

    feature_0 = feature_0.swapaxes(1, 2).swapaxes(2, 3)[0]
    feature_1 = feature_1.swapaxes(1, 2).swapaxes(2, 3)[0]
    feature_2 = feature_2.swapaxes(1, 2).swapaxes(2, 3)[0]
    feature_3 = feature_3.swapaxes(1, 2).swapaxes(2, 3)[0]

    assert len(np.shape(feature_0)) == 3, "The feature shape is {}".format(len(np.shape(feature_0)))
    assert np.shape(feature_0)[-1] == 256, "The last dim is {}".format(np.shape(feature_0)[-1])

    return feature_0, feature_1, feature_2, feature_3


def load_feature_infer(main_path, frame_name):
    frame_path = main_path + str(frame_name) + ".pickle"
    with open(frame_path, "rb") as rf:
        obj = pickle.load(rf)
    curr_dict = obj[list(obj.keys())[0]]
    feature_0, feature_1, feature_2, feature_3 = np.asarray(curr_dict[0]), np.asarray(curr_dict[1]), np.asarray(curr_dict[2]), np.asarray(curr_dict[3])

    feature_0 = feature_0.swapaxes(1, 2).swapaxes(2, 3)
    feature_1 = feature_1.swapaxes(1, 2).swapaxes(2, 3)
    feature_2 = feature_2.swapaxes(1, 2).swapaxes(2, 3)
    feature_3 = feature_3.swapaxes(1, 2).swapaxes(2, 3)

    assert len(np.shape(feature_0)) == 4, "The feature shape is {}".format(len(np.shape(feature_0)))
    assert np.shape(feature_0)[-1] == 256, "The last dim is {}".format(np.shape(feature_0)[-1])

    return feature_0, feature_1, feature_2, feature_3
