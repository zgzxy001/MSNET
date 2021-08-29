import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import sys
sys.path.append("")
from nn_model import define_model
from data_utils import get_trip, load_feature
from model_utils import custom_hinge_loss

from utils import DEFINE_integer
from utils import DEFINE_string

flags = tf.flags
FLAGS = flags.FLAGS

DEFINE_string("trip_pick_path", None, "Path to load trip file")
DEFINE_string("feature_pick_path", None, "Path to load individual feature map")
DEFINE_string("model_save_path", None, "Path to save model")

DEFINE_integer("n_epochs", 10, "How many epochs to train in total")
DEFINE_integer("restore_flag", 1, "Restore or not")
DEFINE_string("model_restore_path", None, "Path to restore model")


def main():

    EPOCHS = FLAGS.n_epochs

    pickle_file = FLAGS.trip_pick_path
    trip_lst = get_trip(pickle_file) # [("5_100", "5_101", "5_200")]


    main_path = FLAGS.feature_pick_path

    with tf.device('/device:GPU:0'):
        x_2 = tf.placeholder(tf.float32, [None, None, 256], name='x')
        pos_2 = tf.placeholder(tf.float32, [None, None, 256], name='pos')
        neg_2 = tf.placeholder(tf.float32, [None, None, 256], name='neg')

        x_3 = tf.placeholder(tf.float32, [None, None, 256], name='x')
        pos_3 = tf.placeholder(tf.float32, [None, None, 256], name='pos')
        neg_3 = tf.placeholder(tf.float32, [None, None, 256], name='neg')

        x_4 = tf.placeholder(tf.float32, [None, None, 256], name='x')
        pos_4 = tf.placeholder(tf.float32, [None, None, 256], name='pos')
        neg_4 = tf.placeholder(tf.float32, [None, None, 256], name='neg')

        x_5 = tf.placeholder(tf.float32, [None, None, 256], name='x')
        pos_5 = tf.placeholder(tf.float32, [None, None, 256], name='pos')
        neg_5 = tf.placeholder(tf.float32, [None, None, 256], name='neg')

        fc_model_train = define_model()

        x_pred_2 = fc_model_train(x_2)
        pos_pred_2 = fc_model_train(pos_2)
        neg_pred_2 = fc_model_train(neg_2)

        x_pred_3 = fc_model_train(x_3)
        pos_pred_3 = fc_model_train(pos_3)
        neg_pred_3 = fc_model_train(neg_3)

        x_pred_4 = fc_model_train(x_4)
        pos_pred_4 = fc_model_train(pos_4)
        neg_pred_4 = fc_model_train(neg_4)

        x_pred_5 = fc_model_train(x_5)
        pos_pred_5 = fc_model_train(pos_5)
        neg_pred_5 = fc_model_train(neg_5)

        loss = custom_hinge_loss(x_pred_2, pos_pred_2, neg_pred_2, x_pred_3,
                                 pos_pred_3, neg_pred_3, x_pred_4, pos_pred_4,
                                 neg_pred_4, x_pred_5, pos_pred_5, neg_pred_5)

        train_op = tf.train.AdamOptimizer().minimize(loss)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if FLAGS.restore_flag:
            fc_model_train.load_weights(FLAGS.model_restore_path)

        for epoch in range(EPOCHS):
            print('Start of epoch %d' % (epoch,))
            total_loss = 0
            for ind, trip in enumerate(trip_lst):

                x_v_2, x_v_3, x_v_4, x_v_5 = load_feature(main_path, trip[0])
                pos_v_2, pos_v_3, pos_v_4, pos_v_5 = load_feature(main_path,
                                                                  trip[1])
                neg_v_2, neg_v_3, neg_v_4, neg_v_5 = load_feature(main_path,
                                                                  trip[2])

                with tf.GradientTape() as tape:
                    _, cur_loss = sess.run([train_op, loss],
                                           feed_dict={x_2: x_v_2,
                                                      pos_2: pos_v_2,
                                                      neg_2: neg_v_2,
                                                      x_3: x_v_3,
                                                      pos_3: pos_v_3,
                                                      neg_3: neg_v_3,
                                                      x_4: x_v_4,
                                                      pos_4: pos_v_4,
                                                      neg_4: neg_v_4,
                                                      x_5: x_v_5,
                                                      pos_5: pos_v_5,
                                                      neg_5: neg_v_5})

                    print('Epoch {} Step {} Loss {:.4f}'.format(epoch, ind, cur_loss))

                    total_loss += cur_loss
                if ind % 1 == 0:
                    fc_model_train.save_weights(
                        FLAGS.model_save_path + "model_{}_{}".format(epoch, ind))
            fc_model_train.save_weights(FLAGS.model_save_path+"model_{}".format(epoch))

            print('All Epoch {} Loss {:.4f}'.format(epoch, total_loss))


if __name__ == "__main__":
    main()

