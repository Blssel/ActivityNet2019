#coding:utf-8

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

"""cfg变量中保存的是默认config，ymal文件保存变动性config
"""
__C = edict()
cfg = __C  # 引用传递

__C.GPUS = '2'  # 末尾无逗号
__C.TRAIN_NO = '000' # 对应log,model存放在logs_0,model_0中

#------Input configuration-------#
__C.INPUT = edict()
__C.INPUT.BASE_INPUT_PATH = '/data1/zhiyuyin/dataset/activitynet'
__C.INPUT.RESTORE_MODEL = False
__C.INPUT.LEN_CLIP = 20  # 左边10秒，右边10秒，相当于大约20*0.5=10秒
__C.INPUT.LEN_CLIP_PRED = 3
__C.INPUT.LEN_INPUT = 2
__C.INPUT.LEN_INPUT_PRED = 2
__C.INPUT.LEN_INPUT_FLOW_PRED = 2
__C.INPUT.LEN_GT = 3
__C.INPUT.RESIZE_HEIGHT = 256
__C.INPUT.RESIZE_WIDTH = 256
__C.INPUT.PATH_TO_GT = '/data1/zhiyuyin/dataset/UCSD/ped2/ped2.mat'
__C.INPUT.PATH_TO_FRAMES = '/data1/zhiyuyin/dataset/UCSD/ped2/testing/frames'


#------Output configuration-------#
__C.OUTPUT = edict()
__C.OUTPUT.BASE_OUTPUTS_PATH = '/data1/zhiyuyin/experiments/ICML/Trail1'
__C.OUTPUT.BASE_OUTPUTS_PATH_FLOW_AE = '/data1/zhiyuyin/experiments/ICML/Trail1/flow_ae'
__C.OUTPUT.BASE_OUTPUTS_PATH_POSE = '/data1/zhiyuyin/experiments/ICML/Trail1/pose'
__C.OUTPUT.BASE_OUTPUTS_PATH_AE = '/data1/zhiyuyin/experiments/ICML/Trail1/autoencoder'
__C.OUTPUT.BASE_OUTPUTS_PATH_ERASE = '/data1/zhiyuyin/experiments/ICML/Trail1/erase'
__C.OUTPUT.BASE_OUTPUTS_PATH_ADVERS = '/data1/zhiyuyin/experiments/ICML/Trail1/advers'
__C.OUTPUT.BASE_OUTPUTS_PATH_PRED_AE = '/data1/zhiyuyin/experiments/ICML/Trail1/pred_ae'
__C.OUTPUT.BASE_OUTPUTS_PATH_PRED_ADV = '/data1/zhiyuyin/experiments/ICML/Trail1/pred_adv'
__C.OUTPUT.BASE_OUTPUTS_PATH_FLOW_ADV = '/data1/zhiyuyin/experiments/ICML/Trail1/flow_adv'
__C.OUTPUT.BASE_OUTPUTS_PATH_FLOW_PRED = '/data1/zhiyuyin/experiments/ICML/Trail1/flow_pred'
__C.OUTPUT.BASE_OUTPUTS_PATH_FLOW_PRED_ADV = '/data1/zhiyuyin/experiments/ICML/Trail1/flow_pred_adv'
__C.OUTPUT.SUMMARY_DIR = 'logs'
__C.OUTPUT.MODEL_DIR = 'models'
__C.OUTPUT.SAVED_MODEL_PATTERN = 'ICML.ckpt'
__C.OUTPUT.PRINT_LOG_INTERVAL = 10
__C.OUTPUT.VALID_SAVE_INTERVAL = 100
__C.OUTPUT.SUM_WRITE_INTERVAL = 5
__C.OUTPUT.MAX_MODELS_TO_KEEP = 100

#------Training configuraton-------#
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 10

__C.TRAIN.LEARNING_RATE_BASE = 0.00001
__C.TRAIN.LEARNING_RATE_BASE_AE = 0.00001
__C.TRAIN.LEARNING_RATE_BASE_ERASE_D = 0.00001
__C.TRAIN.LEARNING_RATE_BASE_ERASE_G = 0.00001
#__C.TRAIN.DECAY_STEP = 300
__C.TRAIN.DECAY_STEP = 100000000
__C.TRAIN.DECAY_RATE = 0.9
__C.TRAIN.LAM_LP = 0.5
__C.TRAIN.LAM_ADV = 0.5
__C.TRAIN.LAM_LP_ERASE = 1.0
__C.TRAIN.LAM_GD_ERASE = 1.0
__C.TRAIN.LAM_ADV_ERASE = 0.05
__C.TRAIN.EPSILON = 0.07


#------Valid configuraton-------#
__C.VALID = edict()
__C.VALID.BATCH_SIZE = 1
__C.VALID.RESTORE_FROM = 'models/SBD_FCN.ckpt-1950'


def get_output_dir(config_file_name):
  """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.
  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.EXP_DIR, osp.basename(config_file_name)))
  if not osp.exists(outdir):
    os.makedirs(outdir)
  return outdir


def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  # for k, v in a.iteritems(): # python2
  for k, v in a.items():
    # a must specify keys that are in b
    # if not b.has_key(k): # python2
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print('Error under config key: {}'.format(k))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)