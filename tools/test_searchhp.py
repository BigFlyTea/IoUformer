import os
import cv2
import numpy as np
from os.path import join
from utils.bbox import get_axis_aligned_bbox
from toolkit.utils.region import vot_overlap, vot_float2str

def track_tune(tracker, config, video, v_idx, tracker_path):
	hp = config['hp']
	benchmark_name = config['benchmark']

	if 'VOT' in benchmark_name:
		baseline_path = join(tracker_path, 'baseline')
		video_path = join(baseline_path, video.name)
		if not os.path.exists(video_path):
			os.makedirs(video_path)
		result_path = join(video_path, video.name + '_001.txt')
	elif 'OTB' in benchmark_name or 'UAV' in benchmark_name or 'TC' in benchmark_name:
		result_path = join(tracker_path, '{:s}.txt'.format(video.name))
	elif 'GOT' in benchmark_name:
		video_path = os.path.join(tracker_path, video.name)
		if not os.path.isdir(video_path):
			os.makedirs(video_path)
		result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
	else:
		raise Exception('benchmark not supported now')

	if benchmark_name in ['VOT2016', 'VOT2018', 'VOT2019']:	
		frame_counter, lost_number, toc = 0, 0, 0
		pred_bboxes = []
		for idx, (img, gt_bbox) in enumerate(video):
			#if len(gt_bbox) == 4:
			#	gt_bbox = [gt_bbox[0], gt_bbox[1],
			#			gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
			#			gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
			#			gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
			tic = cv2.getTickCount()
			if idx == frame_counter:
				cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
				gt_bbox_ = [cx-(w)/2, cy-(h)/2, w, h]
				tracker.init(img, gt_bbox_, hp)
				pred_bbox = gt_bbox_
				pred_bboxes.append([float(1)] if 'VOT' in benchmark_name else gt_bbox)
			elif idx > frame_counter:
				outputs = tracker.track(img)
				pred_bbox = outputs['bbox']
				overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
				if overlap > 0:
					# not lost
					pred_bboxes.append(pred_bbox)
				else:
					# lost object
					pred_bboxes.append([float(2)])
					frame_counter = idx + 5 # skip 5 frames
					lost_number += 1
			else:
				pred_bboxes.append([float(0)])
			toc += cv2.getTickCount() - tic
		toc /= cv2.getTickFrequency()
		print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
					v_idx+1, video.name, toc, idx / toc, lost_number))
	else:
		#OPE tracking
		toc = 0
		pred_bboxes = []
		track_times = []
		print(video.name)
		for idx, (img, gt_bbox) in enumerate(video):
			tic = cv2.getTickCount()
			if idx == 0:
				cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
				gt_bbox_ = [cx-(w)/2, cy-(h)/2, w, h]
				tracker.init(img, gt_bbox_, hp)
				pred_bbox = gt_bbox_
				if 'VOT2018-LT' == benchmark_name:
					pred_bboxes.append([1])
				else:
					pred_bboxes.append(pred_bbox)
			else:
				outputs = tracker.track(img)
				pred_bbox = outputs['bbox']
				pred_bboxes.append(pred_bbox)
			toc += cv2.getTickCount() - tic
			track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
		toc /= cv2.getTickFrequency()
		print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))
	# save results
	if 'VOT' in benchmark_name:
		with open(result_path, 'w') as f:
			for x in pred_bboxes:
				if isinstance(x, int):
					f.write("{:d}\n".format(x))
				else:
					f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
	elif 'OTB' in benchmark_name or 'UAV' in benchmark_name or 'TC' in benchmark_name:
		with open(result_path, 'w') as f:
			for x in pred_bboxes:
				f.write(','.join([str(i) for i in x]) + '\n')
				#p_bbox = x.copy()
				#f.write(
                #    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')
	elif 'GOT' in benchmark_name:
		with open(result_path, 'w') as f:
			for x in pred_bboxes:
				f.write(','.join([str(i) for i in x])+'\n')
			result_path = os.path.join(video_path,
					   '{}_time.txt'.format(video.name))
			with open(result_path, 'w') as f:
				for x in track_times:
					f.write("{:.6f}\n".format(x))
	
	