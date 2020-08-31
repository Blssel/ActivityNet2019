# coding:utf-8
import glob
import os
import cv2
import random
import json
import os.path as osp
import numpy as np
import pandas as pd
import tensorflow as tf

__author__ = 'Zhiyu Yin'


class ActivityNetBatchGenerator(object):
  num_samples = 0
  num_samples_valid = 0

  def __init__(self, cfg):
    self.cfg = cfg
    """为了加速训练，决定把数据加载进内存中"""
    self.vid_boundaries_dicts_train,  self.vid_boundaries_dicts_valid = self._prepare_for_trainval(self.cfg)

  @staticmethod
  def _load_meta_data(cfg):
    path2json = osp.join(cfg.INPUT.BASE_INPUT_PATH, 'activity_net.v1-3.min.json')
    print(path2json)
    with open(path2json) as json_file:
      annotation = json.load(json_file)
    return annotation

  @staticmethod
  def _sec2sampledframeNo(sec, fps, sample_interval):
    return int((sec * fps) // sample_interval)

  def _prepare_for_trainval(self, cfg):
    self.annotation = self._load_meta_data(cfg)  # 解析json，即annotation文件
    annotation = self.annotation
    """构建vid_boundaries_dicts"""
    vid_boundaries_dicts_train = {}  # like{'vid_name1':[0,134,345 , ...], 'vid_name2':[...], ...]
    vid_boundaries_dicts_valid = {}  # like{'vid_name1':[0,134,345 , ...], 'vid_name2':[...], ...]
    database = annotation["database"]
    items = database.items()
    i = -1
    for vid_name, anno in items:
      i += 1
      if i % 1000 == 0:
        print('%d of %d processed' % (i, len(items)))
      if anno['subset'] == 'testing':
        continue
      path_to_feature = osp.join(cfg.INPUT.BASE_INPUT_PATH, "anet_cuhk_original_feature", "csv_action_spatial", "v_" + vid_name + ".csv")
      if not os.path.exists(path_to_feature):
        continue
      """寻找所有边界"""
      vid_boundaries_dicts = vid_boundaries_dicts_train if anno['subset'] == 'training' else vid_boundaries_dicts_valid
      if vid_name not in vid_boundaries_dicts:
        vid_boundaries_dicts[vid_name] = [0]
      if osp.exists(osp.join(cfg.INPUT.BASE_INPUT_PATH, 'ActivityNetVideoData', 'videos', 'v_' + vid_name + '.mp4')):
        vid_cap = cv2.VideoCapture(osp.join(cfg.INPUT.BASE_INPUT_PATH, 'ActivityNetVideoData', 'videos', 'v_' + vid_name + '.mp4'))  # 读视频，获取帧数
      elif osp.exists(osp.join(cfg.INPUT.BASE_INPUT_PATH, 'ActivityNetVideoData', 'videos', 'v_' + vid_name + '.mkv')):
        vid_cap = cv2.VideoCapture(osp.join(cfg.INPUT.BASE_INPUT_PATH, 'ActivityNetVideoData', 'videos', 'v_' + vid_name + '.mkv'))  # 读视频，获取帧数
      elif osp.exists(osp.join(cfg.INPUT.BASE_INPUT_PATH, 'ActivityNetVideoData', 'videos', 'v_' + vid_name + '.webm')):
        vid_cap = cv2.VideoCapture(osp.join(cfg.INPUT.BASE_INPUT_PATH, 'ActivityNetVideoData', 'videos', 'v_' + vid_name + '.webm'))  # 读视频，获取帧数
      else:
        print('###########')
        print(osp.join(cfg.INPUT.BASE_INPUT_PATH, 'ActivityNetVideoData', 'videos', 'v_' + vid_name + '.mp4'))
        print('###########')
        exit(0)
        vid_cap = cv2.VideoCapture(None)
      if not vid_cap.isOpened():
        exit(0)
      num_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
      fps = vid_cap.get(cv2.CAP_PROP_FPS)
      duration = float(anno['duration'])
      # print(duration, '==', 1.0 * num_frames/fps)
      for seg in anno['annotations']:
        begin = seg['segment'][0]
        end = seg['segment'][1]
        frameNo1 = self._sec2sampledframeNo(begin, fps, 16.0)  # 将秒数对应到帧上，再对应到最近的采样帧上
        frameNo2 = self._sec2sampledframeNo(end, fps, 16.0)  # 将秒数对应到帧上，再对应到最近的采样帧上
        if frameNo1 > vid_boundaries_dicts[vid_name][-1]:
          vid_boundaries_dicts[vid_name].append(frameNo1)
        if frameNo2 > vid_boundaries_dicts[vid_name][-1]:
          vid_boundaries_dicts[vid_name].append(frameNo2)
      if self._sec2sampledframeNo(duration, fps, 16.0) > vid_boundaries_dicts[vid_name][-1]:
        vid_boundaries_dicts[vid_name].append(self._sec2sampledframeNo(duration, fps, 16.0))
    return vid_boundaries_dicts_train, vid_boundaries_dicts_valid

  def read(self, mode):
    def video_clip_generator_train():
      clip = []
      gt = []
      vid_boundaries_dicts_train = self.vid_boundaries_dicts_train
      while True:
        for vid_name, boundaries in vid_boundaries_dicts_train:
          path_to_feature_spatial = osp.join(cfg.INPUT.BASE_INPUT_PATH, "anet_cuhk_original_feature", "csv_action_spatial", "v_" + vid_name + ".csv")  #  读入csv
          path_to_feature_temporal = osp.join(cfg.INPUT.BASE_INPUT_PATH, "anet_cuhk_original_feature", "csv_action_temporal", "v_" + vid_name + ".csv")
          df = pd.read_csv(path_to_feature_spatial)
          for i in range(len(vid_boundaries_dicts_train[vid_name])):
            if i == 0 or i == len(vid_boundaries_dicts_train[vid_name]) - 1:
              continue
            elif (vid_boundaries_dicts_train[vid_name][i] - vid_boundaries_dicts_train[vid_name][i-1] <10) \
                  or (vid_boundaries_dicts_train[vid_name][i + 1] - vid_boundaries_dicts_train[vid_name][i] < 10):
              continue
            else:
              clip




    def video_clip_generator_test():
      clip = []
      gt = []
      while True:
        v_id = (v_id + 1) % num_videos
        video_info = video_info_list[v_id]

        video_clip, flow_clip = [], []
        video_end = False
        for frame_id in range(0, video_info['length']):
          if frame_id == video_info['length'] - 1:  # 区别！！！！！
            break
          video_clip.append(self._np_load_frame(video_info['frame'][frame_id], resize_height, resize_width, False))
          video_clip.append(
            self._np_load_frame(video_info['frame'][frame_id + 1], resize_height, resize_width, False))

          video_clip = np.concatenate(video_clip, axis=2)

          if frame_id == video_info['length'] - 2:  # 区别！！！！！
            video_end = True
          yield video_clip, video_end
          video_clip, flow_clip = [], []

    if mode == 'Train':
      dataset_train = tf.data.Dataset.from_generator(generator=video_clip_generator_train,
                                                     output_types=(tf.float32),
                                                     output_shapes=([resize_height, resize_width, 2 * 3]))
      print('generator dataset, {}'.format(dataset_train))
      dataset_train = dataset_train.prefetch(buffer_size=10)
      dataset_train = dataset_train.shuffle(buffer_size=1000).batch(cfg.TRAIN.BATCH_SIZE)
      print('epoch dataset, {}'.format(dataset_train))
      ite_train = dataset_train.make_one_shot_iterator()
      return ite_train
    else:
      dataset_test = tf.data.Dataset.from_generator(generator=video_clip_generator_test,
                                                    output_types=(tf.float32, tf.bool),
                                                    output_shapes=([resize_height, resize_width, 2 * 3], None))
      print('generator dataset, {}'.format(dataset_test))
      dataset_test = dataset_test.prefetch(buffer_size=10)
      dataset_test = dataset_test.batch(cfg.VALID.BATCH_SIZE)
      ite_test = dataset_test.make_one_shot_iterator()
      return ite_test


if __name__ == '__main__':
  def _parse_args():
    parser = argparse.ArgumentParser(description='Train a keypoint regressor.')
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default=None, type=str)
    args = parser.parse_args()
    return args

  # sys.path.append(os.getcwd())
  import sys
  import argparse
  import pprint
  from config import cfg, cfg_from_file, get_output_dir

  args = _parse_args()
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)  # read from args.cfg_file, and integrate into cfg
  pprint.pprint(cfg)
  activitynet = ActivityNetBatchGenerator(cfg)

