import sys
import json
import os
import glob
import ast
from os.path import join, realpath, dirname
import numpy as np


def rect_iou(rects1, rects2, bound=None):
    r"""Intersection over union.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        bound (numpy.ndarray): A 4 dimensional array, denotes the bound
            (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
    """
    assert rects1.shape == rects2.shape
    if bound is not None:
        # bounded rects1
        rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
        rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
        rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
        rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
        # bounded rects2
        rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
        rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
        rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
        rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

    rects_inter = _intersection(rects1, rects2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

    areas1 = np.prod(rects1[..., 2:], axis=-1)
    areas2 = np.prod(rects2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    eps = np.finfo(float).eps
    ious = areas_inter / (areas_union + eps)
    ious = np.clip(ious, 0.0, 1.0)

    return ious

def _intersection(rects1, rects2):
    r"""Rectangle intersection.

    Args:
        rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
        rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height).
    """
    assert rects1.shape == rects2.shape
    x1 = np.maximum(rects1[..., 0], rects2[..., 0])
    y1 = np.maximum(rects1[..., 1], rects2[..., 1])
    x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                    rects2[..., 0] + rects2[..., 2])
    y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                    rects2[..., 1] + rects2[..., 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T
def evaluate(ious, times):
    # AO, SR and tracking speed
    nbins_iou = 101
    ao = np.mean(ious)
    sr = np.mean(ious > 0.5)
    if len(times) > 0:
        # times has to be an array of positive values
        speed_fps = np.mean(1. / times)
    else:
        speed_fps = -1

    # success curve
    # thr_iou = np.linspace(0, 1, 101)
    thr_iou = np.linspace(0, 1, nbins_iou)
    bin_iou = np.greater(ious[:, None], thr_iou[None, :])
    succ_curve = np.mean(bin_iou, axis=0)

    return ao, sr, speed_fps, succ_curve

def eval_auc_tune(result_path, dataset='OTB2015'):
    #list_path = os.path.join(realpath(dirname(__file__)), '../../', 'dataset', dataset + '.json')
    list_path = os.path.join('/media/zgluo/SanDisk/OTB/OTB100_pysot', dataset+'.json')
    annos = json.load(open(list_path, 'r'))
    seqs = list(annos.keys())  # dict to list for py3
    n_seq = len(seqs)
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success_overlap = np.zeros((n_seq, 1, len(thresholds_overlap)))

    for i in range(n_seq):
        seq = seqs[i]
        gt_rect = np.array(annos[seq]['gt_rect']).astype(np.float)
        gt_center = convert_bb_to_center(gt_rect)
        bb = get_result_bb(result_path, seq)
        center = convert_bb_to_center(bb)
        success_overlap[i][0] = compute_success_overlap(gt_rect, bb)
        
    auc = success_overlap[:, 0, :].mean()
    return auc

def eval_ao_tune(result_path, dataset='GOT10K'):
    '''evaluate got10k validation dataset
    '''
    #json_path = os.path.join('/media/zgluo/SanDisk/GOT10K/full_data/val', 'GOT10k_val_v2'+'.json')
    json_path = '/home/zhangxiang/Data/GOT10K/GOT10K_test_private.json'
    annos = json.load(open(json_path, 'r'))
    seqs = list(annos.keys())
    ious = {}
    times = {}
    for seq_name in seqs:
        #bound = ast.literal_eval(annos[seq_name]['resolution'])
        gts = np.array(annos[seq_name]['gt_rect'])
        #covers = np.array(annos[seq_name]['cover'])

        record_files = glob.glob(os.path.join(result_path, seq_name, '%s_[0-9]*.txt' % seq_name))
        if len(record_files) == 0:
            raise Exception('Results for sequence %s not found.' % seq_name)
        # read results of all repetitions
        boxes = [np.loadtxt(f, delimiter=',') for f in record_files]
        seq_ious = [rect_iou(b[1:], gts[1:], bound=None) for b in boxes]
        #seq_ious = [rect_iou(b[1:], gts[1:], bound=bound) for b in boxes]
        # only consider valid frames where targets are visible
        #seq_ious = [t[covers[1:] > 0] for t in seq_ious]
        seq_ious = np.concatenate(seq_ious)
        ious[seq_name] = seq_ious

        # stack all tracking times
        times[seq_name] = []
        time_file = os.path.join(result_path, seq_name, '%s_time.txt' % seq_name)
        if os.path.exists(time_file):
            seq_times = np.loadtxt(time_file, delimiter=',')
            seq_times = seq_times[~np.isnan(seq_times)]
            seq_times = seq_times[seq_times > 0]
            if len(seq_times) > 0:
                times[seq_name] = seq_times
        
        # store sequence-wise performance
        #ao, sr, speed, _ = evaluate(seq_ious, seq_times)
    ious = np.concatenate(list(ious.values()))
    times = np.concatenate(list(times.values()))
    # store overall performance
    ao, sr, speed, succ_curve = evaluate(ious, times)
    print(ao, sr, speed)
    return ao

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('python ./lib/core/eval_got10k.py  ./results/GOT10k/TSSiamFC GOT10K')
        exit()
    result_path = sys.argv[1]
    dataset = sys.argv[2]
    ao = eval_ao_tune(result_path, dataset)
    print(ao)
