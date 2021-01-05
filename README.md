# MSNET
This repository contains the code and models for the following WACV'21 paper:

**[MSNet: A Multilevel Instance Segmentation Network for Natural Disaster Damage Assessment in Aerial Videos](https://arxiv.org/abs/2006.16479)** 

If you find this code useful in your research then please cite

```
@misc{zhu2020msnet,
    title={MSNet: A Multilevel Instance Segmentation Network for Natural Disaster Damage Assessment in Aerial Videos},
    author={Xiaoyu Zhu and Junwei Liang and Alexander Hauptmann},
    year={2020},
    eprint={2006.16479},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
# The ISBDA Dataset

+ Download links: [Google Drive](https://drive.google.com/file/d/1kEKJ8kr1aScXz_1El7Mn-Yi0ANducQIW/view?usp=sharing)

+ The dataset has 1,030 images with 2,961 annotations. Each annotation includes the damage segmentation, damage bounding boxes, and house bounding boxes. 

<div align="center">
  <div style="">
      <img src="data_vis.png"/>
  </div>
</div>

# Code
## Training
```
$ cd ./examples/msnet/
$ python train.py --config DATA.BASEDIR=data_dir MODE_FPN=True \
  DATA.VAL=('val',)  DATA.TRAIN=('train',)  \
  TRAIN.BASE_LR=1e-3 TRAIN.EVAL_PERIOD=1 TRAIN.LR_SCHEDULE=[3000]  \
  PREPROC.TRAIN_SHORT_EDGE_SIZE=[600,1200] TRAIN.CHECKPOINT_PERIOD=1 DATA.NUM_WORKERS=1 \
  --load  checkpoint_dir\
  --logdir log_dir
```
## Inferencing
```
$ cd ./examples/msnet/
$ python predict.py \
 --config DATA.BASEDIR=data_dir MODE_FPN=True \
 DATA.VAL=('val',)  DATA.TRAIN=('train',) \
 --load checkpoint_dir --evaluate output_json_file
```
## Pre-Trained Model
Download links: [Google Drive](https://drive.google.com/drive/folders/1aCvZ-jjYKJbserHrdK3X4xS6tLFGIbpp?usp=sharing)

## Acknowledgements
Our MSNet is based on [Tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN).
