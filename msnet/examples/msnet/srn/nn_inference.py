import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import sys
sys.path.append("")
from nn_model import define_model
from data_utils import load_feature_infer

from utils import DEFINE_integer
from utils import DEFINE_string
from utils import load_val_lst

import pickle

flags = tf.flags
FLAGS = flags.FLAGS

DEFINE_string("trip_pick_path", None, "Path to load trip file")
DEFINE_string("feature_pick_path", None, "Path to load individual feature map")
DEFINE_string("model_load_path", None, "Path to load model")
DEFINE_string("predict_save_path", None, "Path to save prediction")
DEFINE_integer("total_val", 240, "Max validation images")




def save_predicted(feature, path, trip):
    all_path = "{}_{}".format(path, trip)
    with open(all_path, "wb") as f:
        pickle.dump(feature, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    main_path = FLAGS.feature_pick_path
    with tf.device('/device:GPU:0'):

        fc_model_train = define_model()
        fc_model_train.load_weights(
           FLAGS.model_load_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        fc_model_train.load_weights(FLAGS.model_load_path)
        val_lst = load_val_lst(FLAGS.total_val)
        for ind, val_num in enumerate(val_lst):
            x_v_2, x_v_3, x_v_4, x_v_5 = load_feature_infer(main_path, val_num)

            x_pred_2 = fc_model_train.predict(x_v_2)
            x_pred_3 = fc_model_train.predict(x_v_3)
            x_pred_4 = fc_model_train.predict(x_v_4)
            x_pred_5 = fc_model_train.predict(x_v_5)
            x_pred = [x_pred_2, x_pred_3, x_pred_4, x_pred_5]
            save_predicted(x_pred, FLAGS.predict_save_path, val_num)
            print('Save finished {}'.format(val_num))


if __name__ == "__main__":
    main()

