import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from os.path import join

import ray #0.8.7
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from hyperopt import hp
from pprint import pprint


from model_builder import ModelBuilder
from tracker_builder import build_tracker
from toolkit.datasets import DatasetFactory
from test_searchhp import track_tune

from toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, \
        VOTDataset, NFSDataset, VOTLTDataset
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, \
        EAOBenchmark, F1Benchmark
from toolkit.evaluation.eval_otb import eval_auc_tune

parser = argparse.ArgumentParser(description='parameters for Ocean tracker')

parser.add_argument('--cache_dir', default='./TC128_TPE_results', type=str, help='directory to store cache')
parser.add_argument('--gpu_nums', default=1, type=int, help='gpu numbers')
parser.add_argument('--trial_per_gpu', default=1, type=int, help='trail per gpu')
parser.add_argument('--dataset', default='TC128', type=str, help='dataset')

args = parser.parse_args()

def eao_vot(tracker, track_config, dataset):
	hp = track_config['hp']
	benchmark_name = track_config['benchmark']
	tracker_path = join('test', (benchmark_name + '_min_iou_{:f}'.format(hp['min_iou'])
												+ '_boxes_per_frame_{:f}'.format(hp['boxes_per_frame'])
												+ '_sigma_factor_{:f}'.format(hp['sigma_factor'])
												+ '_IOU_THRESHOLD_{:f}'.format(hp['IOU_THRESHOLD'])
												#+ '_w_inf_{:f}'.format(hp['window_influence'])	
												#+ '_search_scale_{:f}'.format(hp['search_scale'])
												#+ '_cg_iter_{:f}'.format(hp['cg_iter'])
												#+ '_online_inf_{:f}'.format(hp['online_influence'])
												#+ '_frame_interval_{:f}'.format(hp['frame_interval'])
													))
	

	if not os.path.exists(tracker_path):
		os.makedirs(tracker_path)
	
	for v_idx, video in enumerate(dataset):
		track_tune(tracker, track_config, video, v_idx, tracker_path)

	re_path = tracker_path.split('/')[0]
	tracker = tracker_path.split('/')[1]
	print(tracker_path)
	dataset.set_tracker(re_path, tracker)
	benchmark = EAOBenchmark(dataset)
	eao = benchmark.eval(tracker)
	eao = eao[tracker]['all']

	return eao

def auc_otb(tracker, track_config, dataset):
	hp = track_config['hp']
	benchmark_name = track_config['benchmark']
	tracker_path = join('test', (benchmark_name + '_min_iou_{:f}'.format(hp['min_iou'])
												+ '_boxes_per_frame_{:f}'.format(hp['boxes_per_frame'])
												+ '_sigma_factor_{:f}'.format(hp['sigma_factor'])
												+ '_IOU_THRESHOLD_{:f}'.format(hp['IOU_THRESHOLD'])
												))
	

	if not os.path.exists(tracker_path):
		os.makedirs(tracker_path)

	for v_idx, video in enumerate(dataset):
		track_tune(tracker, track_config, video, v_idx, tracker_path)
	auc = eval_auc_tune(tracker_path, track_config['benchmark'])
	return auc

def ao_got(tracker, track_config, dataset):
	hp = track_config['hp']
	benchmark_name = track_config['benchmark']
	tracker_path = join('test', (benchmark_name + '_smaller_weight_{:f}'.format(hp['smaller_weight'])
												+ '_larger_weight_{:f}'.format(hp['larger_weight'])
												+ '_smaller_scale_{:f}'.format(hp['smaller_scale'])
												+ '_larger_scale_{:f}'.format(hp['larger_scale'])
												+ '_w_inf_{:f}'.format(hp['window_influence'])	
												+ '_search_scale_{:f}'.format(hp['search_scale'])))
	if not os.path.exists(tracker_path):
		os.makedirs(tracker_path)
	for v_idx, video in enumerate(dataset):
		track_tune(tracker, track_config, video, v_idx, tracker_path)
	ao = eval_ao_tune(tracker_path, track_config['benchmark'])
	return ao
	
def fitness(config, reporter):
	#create model
	model = ModelBuilder()
	#build tracker
	tracker = build_tracker(model)

	if 'OTB' in args.dataset or 'CVPR' in args.dataset:
		dataset_path = '/media/zgluo/SanDisk/OTB/'
		dataset_folder = 'OTB100_pysot'
	elif 'VOT' in args.dataset:
		dataset_path = '/media/zgluo/SanDisk/VOT/'
		dataset_folder = args.dataset
	elif 'UAV' in args.dataset:
		dataset_path = '/home/zhangxiang/Data/'
		dataset_folder = 'UAV123'
	elif 'TC' in args.dataset:
		dataset_path = '/media/zgluo/SanDisk/'
		dataset_folder = 'Temple-color-128'
	#dataset_root = os.path.join(cur_dir, 'testing_dataset', args.dataset)
	dataset_root = os.path.join(dataset_path, dataset_folder)
	dataset = DatasetFactory.create_dataset(name=args.dataset,
											dataset_root=dataset_root,
											load_img=False)
	min_iou = config['min_iou']
	boxes_per_frame = config['boxes_per_frame']
	sigma_factor = config['sigma_factor']
	IOU_THRESHOLD = config['IOU_THRESHOLD']
	#window_inf = config['window_influence']
	#sea_scale = config['search_scale']
	#online_influence = config['online_influence']
	#cg_iter = config['cg_iter']
	#frame_interval = config['frame_interval']


	model_config = dict()
	model_config['benchmark'] = args.dataset
	model_config['hp'] = dict()
	model_config['hp']['min_iou'] = min_iou
	model_config['hp']['boxes_per_frame'] = boxes_per_frame
	model_config['hp']['sigma_factor'] = sigma_factor
	model_config['hp']['IOU_THRESHOLD'] = IOU_THRESHOLD
	#model_config['hp']['window_influence'] = window_inf
	#model_config['hp']['search_scale'] = sea_scale
	#model_config['hp']['online_influence'] = online_influence
	#model_config['hp']['cg_iter'] = cg_iter
	#model_config['hp']['frame_interval'] = frame_interval
	if args.dataset.startswith('VOT'):
		eao = eao_vot(tracker, model_config, dataset)
		print("min_iou: {0}, boxes_per_frame: {1}, sigma_factor: {2}, \
			   IOU_THRESHOLD: {3}, eao: {4}".format(min_iou, boxes_per_frame, sigma_factor, IOU_THRESHOLD, eao))
		reporter(EAO=eao)

	# OTB and Ocean
	if args.dataset.startswith('OTB') or args.dataset.startswith('UAV') or args.dataset.startswith('TC'):
		auc = auc_otb(tracker, model_config, dataset)
		print("min_iou: {0}, boxes_per_frame: {1}, sigma_factor: {2}, \
			   IOU_THRESHOLD: {3},\
			   auc: {4}".format(min_iou, boxes_per_frame, sigma_factor, IOU_THRESHOLD, auc))
		reporter(AUC=auc)

if __name__ == "__main__":
	'''hyper parameter search
	'''
	ray.init(num_gpus=args.gpu_nums, num_cpus=args.gpu_nums*2, object_store_memory=500000000)
	tune.register_trainable("fitness", fitness)
	params = {
			 "min_iou": hp.quniform('min_iou', 0.1, 0.6, 0.001),
			 "boxes_per_frame": hp.quniform('boxes_per_frame', 16, 64, 1),
			 "sigma_factor": hp.quniform('sigma_factor', 0.01, 0.3, 0.0001),
			 "IOU_THRESHOLD": hp.quniform('IOU_THRESHOLD', 0.1, 0.8, 0.001),
			 #"window_influence": hp.quniform('window_influence', 0.2, 0.8, 0.001),#Min-Max scaling
			 #"search_scale": hp.quniform('search_scale', 2.9, 3.5, 0.001),
			 #"online_influence": hp.quniform('online_influence', 0.01, 0.4, 0.001),#Min-Max scaling
			 #"cg_iter": hp.quniform('cg_iter', 3, 10, 1),
			 #'frame_interval': hp.quniform('frame_interval', 1, 15, 1),
			 #"input_size": hp.quniform()
			}
	print('hyper parameter space')
	pprint(params)

	tune_spec = {
		"zp_tune": {
			"run": "fitness",
			"resources_per_trial": {
				"cpu": 1,  # single task cpu num
				"gpu": 1.0 / args.trial_per_gpu,  # single task gpu num
			},
			"num_samples": 10000,  # sample hyperparameters times
			"local_dir": args.cache_dir
		}
	}

	# stop condition for VOT and OTB
	if args.dataset.startswith('VOT'):
		stop = {
			"EAO": 0.6,  # if EAO >= 0.6, this procedures will stop
			# "timesteps_total": 100, # iteration times
		}
		tune_spec['zp_tune']['stop'] = stop

		scheduler = AsyncHyperBandScheduler(
			# time_attr="timesteps_total",
			metric='EAO',
			mode='max',
			max_t=400,
			grace_period=20
		)
		# max_concurrent: the max running task
		#algo = HyperOptSearch(params, max_concurrent=args.gpu_nums*args.trial_per_gpu + 1, reward_attr="EAO")
		algo = HyperOptSearch(params, max_concurrent=args.gpu_nums*args.trial_per_gpu + 1, metric='EAO',  mode='max')

	elif args.dataset.startswith('OTB') or args.dataset.startswith('UAV') \
		or args.dataset.startswith('GOT10K') or args.dataset.startswith('TC'):
		stop = {
			# "timesteps_total": 100, # iteration times
			"AUC": 0.80
		}
		tune_spec['zp_tune']['stop'] = stop
		scheduler = AsyncHyperBandScheduler(
			# time_attr="timesteps_total",
			metric="AUC",
			mode='max',
			max_t=400,
			grace_period=20
		)
		algo = HyperOptSearch(params, max_concurrent=args.gpu_nums*args.trial_per_gpu + 1, metric="AUC")  #
	else:
		raise ValueError("not support other dataset now")

	tune.run_experiments(tune_spec, search_alg=algo, scheduler=scheduler)
