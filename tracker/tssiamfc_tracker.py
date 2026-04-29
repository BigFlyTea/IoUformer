import cv2
import math
import torch
import importlib
import numpy as np
import torch.nn.functional as F
from scipy import signal
from config import cfg
#from utils.visdom import Visdom
from tracker.base_tracker import SiameseTracker
from IoUnet.lib.tracker.transformer_iou import Transformer_iou
torch.set_printoptions(precision=10)

class TSSiamFCTracker(SiameseTracker):
	"""major track logic
	"""
	def __init__(self, model):
		super(TSSiamFCTracker, self).__init__()
		#self.config = cfg
		self.model = model
		if cfg.CUDA and torch.cuda.is_available():
			self.model.cuda()
		else:
			self.model.cpu()

		self.model.eval()
		#model size 204 is better than 180, 244 is worser.
		self.model_sz = cfg.MODEL.INPUT_SIZE
		
		### init iounet
		iou_parameter = self.get_parameters()
		self.iounet = Transformer_iou(iou_parameter)

	def get_parameters(self):
		"""Get parameters."""
		param_module = importlib.import_module('IoUnet.parameter.{}'.format('transformer_iou'))
		params = param_module.parameters('baseline_got10k_only')
		return params

	def _init_visdom(self, visdom_info, debug):
		visdom_info = {} if visdom_info is None else visdom_info
		self.pause_mode = False
		self.step = False
		if debug > 0 and visdom_info.get('use_visdom', True):
			try:
				self.visdom = Visdom(debug, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
									 visdom_info=visdom_info)

				# Show help
				help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
							'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
							'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
							'block list.'
				self.visdom.register(help_text, 'text', 1, 'Help')
			except:
				time.sleep(0.5)
				print('!!! WARNING: Visdom could not start, so using matplotlib visualization instead !!!\n'
					  '!!! Start Visdom in a separate terminal window by typing \'visdom\' !!!')
		else:
			self.visdom = None
	def _visdom_ui_handler(self, data):
		if data['event_type'] == 'KeyPress':
			if data['key'] == ' ':
				self.pause_mode = not self.pause_mode

			elif data['key'] == 'ArrowRight' and self.pause_mode:
				self.step = True	

	def sea_window_size(self, target_size:float, image_size:tuple, search_scale:int, wh:int) -> float:
		'''calculates the size of the search window
		'''
		search_size = target_size * search_scale
		#max_side = np.max(wh)
		#if search_size<max_side:
		#	search_size = max_side + 0.1*max_side
		#check whether search size out of image, 
		#however, this change may cause bbox out of feature (GOT-10k_Test_000137)!
		if search_size > min(image_size):
			search_size = np.array(min(image_size))

		#search_size = search_size + total_stride * 2 - (search_size - total_stride) % (total_stride * 2)
		#while search_size%(total_stride*2) != 4:
		#	search_size += 1
		
		return search_size

	def sea_window_location(self, target_location:np.ndarray, search_size:float) -> 'np.float:[x,y,sw,sh]':
		"""calculates search window location
		"""
		#srch_window_position = np.floor(target_location[0:2]+target_location[2:4]/2 - search_size/2)
		srch_window_position = (target_location[0:2]+target_location[2:4]/2 - search_size/2)
		srch_window_size = np.array([search_size, search_size])
		srch_window_location = np.append(srch_window_position, srch_window_size)
		assert(srch_window_location.shape == (4,)), "the shape of srch_window_location is {}".format(srch_window_location.shape)
		return srch_window_location

	def init(self, img:'BGR image', bbox: 'list:[x,y,w,h]', hp=None, visdom_info=None, debug=0):
		'''initialzie with first image
		'''
		self._init_visdom(visdom_info, debug)
		self.model.visdom = self.visdom

		if hp:
			cfg.TEST.IOU_THRESHOLD = hp['IOU_THRESHOLD']
			cfg.TEST.min_iou = hp['min_iou']
			cfg.TEST.boxes_per_frame = int(hp['boxes_per_frame'])
			cfg.TEST.sigma_factor =hp['sigma_factor']

		ori_target_size = np.sqrt(bbox[2]*bbox[3])
		
		ori_image_size =img.shape[0:2][::-1]
		#enlarge smaller object
		self.reflag = False
		self.rescale = 1
		if ori_target_size > cfg.MODEL.MAX_SIZE:
			self.reflag = True
			self.rescale = cfg.MODEL.MAX_SIZE / ori_target_size
		if ori_target_size < cfg.MODEL.MIN_SIZE:
			self.reflag = True
			self.rescale = cfg.MODEL.MIN_SIZE / ori_target_size
		if self.reflag:
			img = cv2.resize(
					img,
					tuple((np.ceil(np.array(ori_image_size) * self.rescale)).astype(int)),
					interpolation=cv2.INTER_LINEAR
			)

		image_size = img.shape[0:2]
		
		#change 1-index to 0-index
		self.target_bbox = np.array(bbox)*self.rescale- np.array([1,1,0,0])
		target_size = np.sqrt(self.target_bbox[2]*self.target_bbox[3])

		search_size = self.sea_window_size(target_size, image_size, cfg.MODEL.SEARCH_SCALE, self.target_bbox[2:4])
		#self.input_size = np.array([self.mode_sz, self.model_sz])
		
		self.sea_window_loc = self.sea_window_location(self.target_bbox, search_size)

		sea_win_img = self.get_subwindow_v2(img, self.sea_window_loc, self.model_sz)
		# the another version of get subwindow #
		#self.channel_average = np.mean(image, axis=(0, 1))
		#sea_win_img = self.get_subwindow(image, self.sea_window_loc[0:2],
		#								 self.input_size[0], 
		#								 self.sea_window_loc[2:4],
		#								 self.channel_average)
		scale = self.model_sz / search_size
		re_traget_bbox = self.target_bbox*scale
		init_response = self.model.template(sea_win_img, re_traget_bbox)
		self.init_apce = self.APCE(init_response)
		self.frame_num = 0

		### init with first bb
		self.iounet.initialize(img, self.target_bbox, hp)

	def track(self, img:'BGR image') -> 'ndarray:[x,y,w,h]':
		self.frame_num += 1
		if self.reflag:
			img = cv2.resize(
				img,
				tuple((np.ceil(np.array(img.shape[0:2][::-1]) * self.rescale)).astype(int)),
				interpolation=cv2.INTER_LINEAR
				)
		sea_win_img = self.get_subwindow_v2(img, self.sea_window_loc, self.model_sz)
		#sea_win_img = self.get_subwindow(image, self.sea_window_loc[0:2],
		#								 self.input_size[0], 
		#								 self.sea_window_loc[2:4],
		#								 self.channel_average)
		responses = self.model.track(sea_win_img)
		#ori_responses = responses.new_tensor(responses)
		ori_responses = responses.clone().detach()
		###bilinear and bicubid has about 1% derivation
		responses = F.interpolate(responses, tuple(self.sea_window_loc[-2:].astype(int)), mode='bilinear', align_corners=True)
		
		responses = torch.squeeze(responses)
		window = self.hann2d(torch.Tensor(self.sea_window_loc[-2:].astype(int)), responses.shape[0])
		#scale_responses = scale_responses + cfg.TEST.WINDOW_INFLUENCE*window
		cfg.TEST.SCALE_INFLUENCE = cfg.TEST.WINDOW_INFLUENCE
		cfg.TEST.POSITION_INFLUENCE = cfg.TEST.WINDOW_INFLUENCE
		scale_responses = (1-cfg.TEST.SCALE_INFLUENCE)*responses + cfg.TEST.SCALE_INFLUENCE*window
		scale_idx = self.select_scale(scale_responses, cfg.MODEL.SCALE_WEIGHTS)
		apce = self.APCE(ori_responses[scale_idx])
		#select_response = scale_responses[scale_idx,:,:].cpu().numpy()
		select_scale = cfg.MODEL.SCALES[scale_idx]
		
		position_response = (1-cfg.TEST.POSITION_INFLUENCE)*responses[scale_idx] + cfg.TEST.POSITION_INFLUENCE*window[scale_idx]
		position_response = position_response.detach().cpu().numpy()
		'''
		max_y, max_x = np.where(position_response == np.max(position_response))
		#there are may multi max places
		if len(max_y)>1:
			max_y = np.array([max_y[0],])
		if len(max_x)>1:
			max_x = np.array([max_x[0],])
		'''
		curr = np.unravel_index(np.argmax(position_response, axis=None),position_response.shape)

		#update new target_bbox
		##calculate displacement in [cx, cy, w, h]
		target_bbox_center = np.append(self.target_bbox[0:2]+self.target_bbox[2:4]/2, self.target_bbox[2:4])
		#search window height and width is 1-index, while position is 0-index
		displacement = (np.append(curr[1], curr[0]) - self.sea_window_loc[2:4]/2+1)*select_scale
		target_bbox_center[0:2] = target_bbox_center[0:2] + displacement
		
		target_bbox_center[2:4] = target_bbox_center[2:4]*select_scale



		##change to [x, y, w, h]
		self.target_bbox = np.append(target_bbox_center[0:2] - target_bbox_center[2:4]/2, target_bbox_center[2:4])
		
		### refine with iou guide
		refine_bb = self.iounet.track(img, self.target_bbox)
		#refine_bb = self.target_bbox
		##update seach window
		self.sea_window_loc[2:4] = self.sea_window_loc[2:4]*select_scale
		self.sea_window_loc[0:2] = target_bbox_center[0:2] - self.sea_window_loc[2:4]/2

		## update sequence label with displacement
		#displacement is in original image space, while label is in deep space
		## update method 1
		#if self.frame_num % cfg.TEST.FRAME_INTERVAL == 0 and cfg.TEST.ONLINE:
		## update method 2
		if cfg.TEST.ONLINE and self.init_apce*cfg.TEST.APCE_HIGH_THRESHOLD>apce\
									>self.init_apce*cfg.TEST.APCE_LOW_THRESHOLD:
			deep_dis = displacement*(self.model_sz/self.sea_window_loc[2:4])/cfg.MODEL.TOTAL_STRIDE
			self.model.update(scale_idx, deep_dis)
			#self.model.update_net(scale_idx, deep_dis)
		search_bbox = (self.sea_window_loc + np.array([1,1,0,0]))/self.rescale
		target_predict = (refine_bb + np.array([1,1,0,0]))/self.rescale
		return {
				'bbox': target_predict,
				'seach_window': search_bbox
				}

	def hann1d(self, sz: int, centered = True) -> torch.Tensor:
		"""1D cosine window."""
		if centered:
			return 0.5 * (1 - torch.cos((2 * math.pi / (sz + 1)) * torch.arange(1, sz + 1).float()))
		w = 0.5 * (1 + torch.cos((2 * math.pi / (sz + 2)) * torch.arange(0, sz//2 + 1).float()))
		return torch.cat([w, w[1:sz-sz//2].flip((0,))])

	def hann2d(self, sz: torch.Tensor, channel:'three scale', centered = True) -> torch.Tensor:
		"""2D cosine window."""
		window = self.hann1d(sz[0].item(), centered).reshape(1, -1, 1) * self.hann1d(sz[1].item(), centered).reshape(1, 1, -1) 
		window = window.repeat(channel,1,1)
		if cfg.CUDA and torch.cuda.is_available():
			window = window.cuda()
		else:
			window = window.cpu()
		return window

	def APCE(self, response_map):
		if isinstance(response_map, torch.Tensor):
			response_map = response_map.cpu().numpy()
		Fmax=np.max(response_map)
		Fmin=np.min(response_map)
		apce=(Fmax-Fmin)**2/(np.mean((response_map-Fmin)**2))
		return apce

	def PSR(self, response):
		response_map=response.copy()
		max_loc=np.unravel_index(np.argmax(response_map, axis=None),response_map.shape)
		y,x=max_loc
		F_max = np.max(response_map)
		response_map[y-5:y+6,x-5:x+6]=0.
		mean=np.mean(response_map[response_map>0])
		std=np.std(response_map[response_map>0])
		psr=(F_max-mean)/std
		return psr

	def select_scale(self, scaled_maps:torch.tensor, scale_weights:list) -> int:
		'''select scale based on responses and correspoding weights
		'''
		num_scale = len(scale_weights)
		maps = scaled_maps.view(num_scale,-1)
		max_response = torch.max(maps, dim =1)[0]
		if cfg.CUDA and torch.cuda.is_available():
			scale_weights = torch.as_tensor(scale_weights, dtype=torch.float32).cuda()
		else:
			scale_weights = torch.as_tensor(scale_weights, dtype=torch.float32).cpu()
		max_response = max_response*scale_weights
		scale_ind = torch.argmax(max_response)

		return scale_ind
