
from yacs.config import CfgNode as CN
from os.path import join
_C = CN()
cfg = _C
_C.CUDA = True

_C.MODEL = CN()
_C.MODEL.INPUT_SIZE = 132
_C.MODEL.TARGET_SIZE = 59
_C.MODEL.MAX_SIZE = 59
_C.MODEL.MIN_SIZE = 44
#_C.MODEL.MAX_SIZE = 73
#_C.MODEL.MIN_SIZE = 73
_C.MODEL.SEARCH_SCALE = 3.13 #4 is worser than 3? not the larger the better.
#3.15>3.1,3.17>3.2~3.16~2.9>3.3~3

_C.MODEL.SCALES = [0.96429, 1, 1.08219]
_C.MODEL.SCALE_WEIGHTS = [0.98346, 1, 1.00962]

#_C.MODEL.SCALES = [43/45, 1, 47/45]
#_C.MODEL.SCALE_WEIGHTS = [0.98901, 1, 1.00485]


_C.MODEL.TOTAL_STRIDE = 4

_C.BACKBONE = CN()
_C.BACKBONE.VGG16 = CN()
_C.BACKBONE.VGG16.DEPTH = 16
_C.BACKBONE.VGG16.WITH_BN = False
_C.BACKBONE.VGG16.WITH_POOLS = (True,True,False,False)
_C.BACKBONE.VGG16.NUM_STAGES = 4
_C.BACKBONE.VGG16.DILATIONS = (1,1,1,1)
_C.BACKBONE.VGG16.FROZEN_STAGE = 4
_C.BACKBONE.VGG16.BN_EVAL = False
_C.BACKBONE.VGG16.BN_FROZEN = False
_C.BACKBONE.VGG16.CEIL_MODEL = False
#_C.BACKBONE.VGG16.OUT_INDICES = ['conv4_1', 'conv4_3']
_C.BACKBONE.VGG16.OUT_INDICES = ['conv4_3']
_C.BACKBONE.VGG16.PRETRAIN_MAT = join('/media/zgluo/laboratory/pytorch/TADT/TADT-ORI','imagenet-vgg-verydeep-16.mat')

_C.TEST = CN()
_C.TEST.TYPE = 'TSSiamFC'
_C.TEST.WINDOW_INFLUENCE = 0.489
_C.TEST.SCALE_INFLUENCE = 0.489
_C.TEST.POSITION_INFLUENCE = 0.489
#_C.TESTDATA.seq_base_path = "/home/huayue/pytorch/examples-master/Tracking/OTB2013/"
_C.TEST.bbox_output_path = "./result/"

_C.TEST.ONLINE = True
_C.TEST.FRAME_INTERVAL = 13
_C.TEST.APCE_LOW_THRESHOLD = 0.44113
_C.TEST.APCE_HIGH_THRESHOLD = 0.55571
_C.TEST.CG_ITER = 10
_C.TEST.ONLINE_INFLUENCE = 0.517
_C.TEST.NORM_THRESHOLD = 0.355

_C.TEST.IOU_THRESHOLD = 0.4
_C.TEST.min_iou = 0.4
_C.TEST.boxes_per_frame = 18
_C.TEST.sigma_factor = 0.05

_C.TEST.filter_init_method = 'ones'
_C.TEST.projection_activation = 'relu'
_C.TEST.response_activation = 'relu'

_C.TEST.OTB_PATH = '/media/zgluo/SanDisk/OTB/'
_C.TEST.VOT_PATH = '/media/zgluo/SanDisk/VOT/'
_C.TEST.UAV_PATH = '/media/zgluo/SanDisk/'
_C.TEST.TColor_PATH = '/media/zgluo/SanDisk/'
_C.TEST.GOT_PATH = '/media/zgluo/SanDisk/GOT10K/full_data/'