
import os
import sys
import cv2
import time
import torch
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

from configs.default import cfg
from toolkit.datasets import DatasetFactory
from model_builder import ModelBuilder
from tracker_builder import build_tracker
from utils.bbox import get_axis_aligned_bbox
from toolkit.utils.region import vot_overlap, vot_float2str

parser = argparse.ArgumentParser(description='tssiamfc tracking')
parser.add_argument('--dataset', default='VOT2016', type=str,
		help='datasets')
parser.add_argument('--config', default='configs/default.py', type=str,
		help='config file')
parser.add_argument('--snapshot', default='', type=str,
		help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
		help='eval one special video')
parser.add_argument('--vis', action='store_true',
		help='whether visualzie result')
parser.add_argument('--debug', type=int, default=0, help='Debug level.')
parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom.')
parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom.')
parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom.')
args = parser.parse_args()
	


def main():
	#load config
	cfg.merge_from_file(args.config)
	
	if not torch.cuda.is_available():
		print('GPU IS NOT AVAILABLE')
	elif cfg.CUDA:
		print('GPU {} is used'.format(torch.cuda.get_device_name(0)))

	#load test data
	#cur_dir = os.path.dirname(os.path.realpath(__file__))
	if 'OTB' in args.dataset or 'CVPR' in args.dataset:
		dataset_path = cfg.TEST.OTB_PATH
		dataset_folder = 'OTB100_pysot'
	elif 'VOT' in args.dataset:
		dataset_path = cfg.TEST.VOT_PATH
		dataset_folder = args.dataset
	elif 'GOT' in args.dataset:
		dataset_path = '/media/zgluo/SanDisk/GOT10K/full_data/'
		dataset_folder = 'val'
	elif 'UAV' in args.dataset:
		dataset_path = '/media/zgluo/SanDisk/'
		dataset_folder = 'UAV123'
	#dataset_root = os.path.join(cur_dir, 'testing_dataset', args.dataset)
	dataset_root = os.path.join(dataset_path, dataset_folder)
	dataset = DatasetFactory.create_dataset(name=args.dataset,
											dataset_root=dataset_root,
											load_img=False)
	#create model
	model = ModelBuilder()

	#build visdom message, init visdom based on every sequence
	visdom_info = {'debug':args.debug, 'use_visdom': args.use_visdom, \
				  'server': args.visdom_server, 'port': args.visdom_port}
	#build tracker
	tracker = build_tracker(model)
	model_name = cfg.TEST.TYPE

	total_lost = 0
	if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
		# restart tracking
		for v_idx, video in enumerate(dataset):
			if args.video != '':
				# test one special video
				if video.name != args.video:
					continue
			frame_counter = 0
			lost_number = 0
			toc = 0
			pred_bboxes = [] #1: init, 2: lost, 0:skip
			for idx, (img, gt_bbox) in enumerate(video):
				if len(gt_bbox) == 4:
					gt_bbox = [gt_bbox[0], gt_bbox[1],
						gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
						gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
						gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
				tic = cv2.getTickCount()
				if idx == frame_counter:
					cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
					gt_bbox_ = [cx-(w)/2, cy-(h)/2, w, h]
					tracker.init(img, gt_bbox_, visdom_info=visdom_info, debug=args.debug)
					pred_bbox = gt_bbox_
					pred_bboxes.append(1)
				elif idx > frame_counter:
					outputs = tracker.track(img)
					pred_bbox = outputs['bbox']
					overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
					if overlap > 0:
						# not lost
						pred_bboxes.append(pred_bbox)
					else:
						# lost object
						pred_bboxes.append(2)
						frame_counter = idx + 5 # skip 5 frames
						lost_number += 1
				else:
					pred_bboxes.append(0)
				toc += cv2.getTickCount() - tic
				if tracker.visdom is not None:
					while True:
						if not tracker.pause_mode:
							break
						elif tracker.step:
							tracker.step = False
							break
						else:
							time.sleep(0.1)
					tracker.visdom.register((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), list(map(int, pred_bbox))), 'Tracking', 1, 'Tracking')
				if idx == 0:
					cv2.destroyAllWindows()
				if args.vis and idx > frame_counter:
					cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
							True, (0, 255, 0), 3)
					bbox = list(map(int, pred_bbox))
					cv2.rectangle(img, (bbox[0], bbox[1]),
									  (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
					cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
					cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
					
					cv2.imshow(video.name, img)
					cv2.waitKey(1)
			toc /= cv2.getTickFrequency()
			# save results
			video_path = os.path.join('results', args.dataset, model_name,
					'baseline', video.name)
			if not os.path.isdir(video_path):
				os.makedirs(video_path)
			result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
			with open(result_path, 'w') as f:
				for x in pred_bboxes:
					if isinstance(x, int):
						f.write("{:d}\n".format(x))
					else:
						f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
			print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
					v_idx+1, video.name, toc, idx / toc, lost_number))
			total_lost += lost_number
		print("{:s} total lost: {:d}".format(model_name, total_lost))
	else:
		#OPE tracking
		for v_idx, video in enumerate(dataset):
			if args.video != '':
				# test one special video
				if video.name != args.video:
					continue
			toc = 0
			pred_bboxes = []
			track_times = []
			print(video.name)
			for idx, (img, gt_bbox) in enumerate(video):
				tic = cv2.getTickCount()
				if idx == 0:
					cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
					gt_bbox_ = [cx-(w)/2, cy-(h)/2, w, h]
					tracker.init(img, gt_bbox_, visdom_info=visdom_info, debug=args.debug)
					pred_bbox = gt_bbox_
					if 'VOT2018-LT' == args.dataset:
						pred_bboxes.append([1])
					else:
						pred_bboxes.append(pred_bbox)
				else:
					outputs = tracker.track(img)
					pred_bbox = outputs['bbox']
					pred_bboxes.append(pred_bbox)
				toc += cv2.getTickCount() - tic
				track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
				if tracker.visdom is not None:
					while True:
						if not tracker.pause_mode:
							break
						elif tracker.step:
							tracker.step = False
							break
						else:
							time.sleep(0.1)
					tracker.visdom.register((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), list(map(int, pred_bbox))), 'Tracking', 1, 'Tracking')
				#if idx == 0:
				#	cv2.destroyAllWindows()
				if args.vis and idx > 0:
					#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
					gt_bbox = list(map(int, gt_bbox))
					pred_bbox = list(map(int, pred_bbox))
					cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
								  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
					cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
								  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
					cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
					cv2.imwrite(str(idx)+'.jpg', img)
					cv2.imshow(video.name, img)
					cv2.waitKey(1)
			cv2.destroyAllWindows()
			toc /= cv2.getTickFrequency()
			# save results
			if 'VOT2018-LT' == args.dataset:
				video_path = os.path.join('results', args.dataset, model_name,
						'longterm', video.name)
				if not os.path.isdir(video_path):
					os.makedirs(video_path)
				result_path = os.path.join(video_path,
						'{}_001.txt'.format(video.name))
				with open(result_path, 'w') as f:
					for x in pred_bboxes:
						f.write(','.join([str(i) for i in x])+'\n')
				result_path = os.path.join(video_path,
						'{}_001_confidence.value'.format(video.name))
				with open(result_path, 'w') as f:
					for x in scores:
						f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
				result_path = os.path.join(video_path,
						'{}_time.txt'.format(video.name))
				with open(result_path, 'w') as f:
					for x in track_times:
						f.write("{:.6f}\n".format(x))
			elif 'GOT10k' == args.dataset:
				video_path = os.path.join('results', args.dataset, model_name, video.name)
				if not os.path.isdir(video_path):
					os.makedirs(video_path)
				result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
				with open(result_path, 'w') as f:
					for x in pred_bboxes:
						f.write(','.join([str(i) for i in x])+'\n')
				result_path = os.path.join(video_path,
						'{}_time.txt'.format(video.name))
				with open(result_path, 'w') as f:
					for x in track_times:
						f.write("{:.6f}\n".format(x))
			else:
				model_path = os.path.join('results', args.dataset, model_name)
				if not os.path.isdir(model_path):
					os.makedirs(model_path)
				result_path = os.path.join(model_path, '{}.txt'.format(video.name))
				with open(result_path, 'w') as f:
					for x in pred_bboxes:
						f.write(','.join([str(i) for i in x])+'\n')
			print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
				v_idx+1, video.name, toc, idx / toc))

if __name__ == '__main__':
	main()
