import json
import pickle

from box_align import multilevel_roi_align
import tensorflow as tf


from utils import DEFINE_integer
from utils import DEFINE_string




flags = tf.flags
FLAGS = flags.FLAGS

DEFINE_string("anno_file_path", "instances_val.json", "Path to load anno file")
DEFINE_string("rst_file_path", "inner_rst_ori.json", "Path to load rst file")
DEFINE_string("roi_path", "inner_rst_ori.json", "Path to load rst file")

DEFINE_string("feature_pred_path", None, "Path to load predicted feature map")

DEFINE_integer("start_img", 0, "Start image num")


def read_img_dict(path):
    """

    :return: {image_id: image_file_name}
    """
    anno_file = path
    with open(anno_file) as file:
        anno_data = json.load(file)

    # anno_data = json.load(anno_file)
    imgs_lst = anno_data['images']
    imgs_dict = {}
    for img_pair in imgs_lst:
        imgs_dict[img_pair['id']] = img_pair['file_name']
    return imgs_dict


def load_pred(path, img_id):
    with open(path+"_"+str(img_id), "rb") as f:
        obj = pickle.load(f)
    return obj


def roi_rsts(imgs_dict, rst_file_path, pred_path, roi_path):

    with tf.Session() as sess:
        rst_file = rst_file_path
        with open(rst_file) as file:
            rst_data = json.load(file)
        new_parsed_dict = {}
        max_id = -1
        img_id_lst = []
        for rst_dict in rst_data:
            # print("rst_dict = ", rst_dict)
            img_id = rst_dict['image_id']
            img_id_lst.append(img_id)
            if img_id > max_id:
                max_id = img_id
            if img_id in new_parsed_dict:
                new_parsed_dict[img_id].append(rst_dict)
            else:
                new_parsed_dict[img_id] = [rst_dict]

        for img_id, curr_rst_lst in new_parsed_dict.items():
            print("img_id = ", img_id)
            if img_id <= FLAGS.start_img:
                continue
            ori_box_lst = []
            for rst_dict in curr_rst_lst:
                ori_box_lst.append(rst_dict['ori_bbox'])

            img_id_pred = load_pred(pred_path, img_id)

            img_fm = multilevel_roi_align(img_id_pred, ori_box_lst, 7).eval()

            with open(roi_path + str(img_id) + ".pickle", "wb") as f:
                pickle.dump(img_fm, f, protocol=pickle.HIGHEST_PROTOCOL)
    sess.close()


def main():
    imgs_dict = read_img_dict(FLAGS.anno_file_path)
    roi_rsts(imgs_dict, FLAGS.rst_file_path, FLAGS.feature_pred_path, FLAGS.roi_path)

if __name__ == "__main__":
    main()


