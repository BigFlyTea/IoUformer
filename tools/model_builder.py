import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from configs.default import cfg
from backbone.vgg import build_vgg16
from classifier.optim import ConvProblem, FactorizedConvProblem, FactorizedConvProblem_v2
from classifier.libs import TensorList
from classifier.libs.optimization import GaussNewtonCG, ConjugateGradient, GradientDescentL2,  \
	NewtonCG, GradientDescent
class ModelBuilder(nn.Module):
	def __init__(self):
		super(ModelBuilder, self).__init__()

		# build backbone
		self.backbone = build_vgg16(cfg)

		#scale search pad
		self.pad = 2
		self.margin_pad = int(self.pad/2)
		self.losses = torch.zeros(0)
		self.new_losses = torch.zeros(0)

	def channel_selction(self, tensor):
		#tensor = torch.squeeze(tensor)
		batch, channel, width, height = tensor.size()
		re_tensor = tensor.reshape(channel, -1)
		sim = torch.matmul(re_tensor, re_tensor.transpose(0,1)).cpu()
		#diag_ones = torch.diag(torch.ones(channel), diagonal=0)
		zero_tensor = torch.zeros(width*height).cuda()
		#mask = 1 - diag_ones

		#measure = torch.norm((sim-diag_ones)*mask)
		index = np.where(sim.cpu().numpy()==0)
		#ind = torch.where(sim==0)
		count = []
		for i in range(channel):
			count.append(len(np.where(index[0]==i)[0]))
			if torch.equal(re_tensor[i], zero_tensor):
				count[i] = 0
			#print(len(np.where(index[0]==i)[0]))
		idx = np.where(np.array(count)>11)[0]
		#print(len(np.where(np.array(count)>50)[0]))
		return idx

	def cos_window(self, sz):
	    cos_window = np.hanning(int(sz[0]))[:, np.newaxis].dot(np.hanning(int(sz[1]))[np.newaxis, :])
	    cos_window = self.ori_label.new_tensor(cos_window)
	    return cos_window

	def template(self, z, bbox):
		with torch.no_grad():
			zfs = self.backbone(z)
		target_size = bbox[2:4]

		crop = 0
		patch_features = []
		for zf in zfs:
			zf_size = torch.tensor(zf.shape[-2:]).numpy()
			center = zf_size/2
			#note the inconsistency between bbox and feature shape
			#bbox[2:4] is [w, h], feature shape is (B,C,H,W)
			patch_size = target_size[::-1]/cfg.MODEL.TOTAL_STRIDE
			#assert (patch_size<zf_size).all(), 'bbox out of feature'
			#the center position is C.5, and target size is h.x, w.x
			#np.ceil(c.5+(h.x,w.x))-np.floor(c.5-(h.x,w.x)) is odd
			patch_loc = np.append(np.floor(center - patch_size/2), np.ceil(center + patch_size/2)).astype(int)
			patch_fea = zf[:,:,patch_loc[0]-crop:patch_loc[2]+crop,patch_loc[1]-crop:patch_loc[3]+crop]
			patch_features.append(patch_fea)

		self.zf = torch.cat(patch_features, dim=1)
		self.select_idx = self.channel_selction(self.zf)
		self.zf = self.zf[:,self.select_idx,:,:]

		h, w = self.zf.shape[-2:]
		#assert h%2 == 1, 'target-specific kernel height is even'
		#assert w%2 == 1, 'target-specific kernel width is even'

		padding = (np.floor([h/2, w/2]).astype(int))
		self.p2d = (padding[1], padding[1], padding[0], padding[0])

		
		## normalization 1
		#std, mean = torch.std_mean(self.zf)
		#zf_v2 = (self.zf-mean)/std
		# this normlization has bad effect if exists negative value.
		#self.zf_v2 = (zf_v2 - torch.min(zf_v2))/zf_v2.numel()

		## normalization 2
		# normalization has no influence on performance.
		nor_zf = self.zf.view(1, -1)		
		nor_zf = F.normalize(nor_zf, dim=1)/nor_zf.numel()
		self.nor_zf = nor_zf.reshape(1, -1, h, w)
		zfs = torch.cat(zfs, dim=1)
		sample = zfs.clone().detach().requires_grad_(False)[:,self.select_idx,:,:]
		kernel = self.nor_zf.clone().detach().requires_grad_(False)
		self.ori_label = F.conv2d(sample, kernel).squeeze()
		ori_label = self.ori_label.cpu().numpy()
		'''
		max_y, max_x = np.where(ori_label == np.max(ori_label))
		l_h, l_w = self.ori_label.shape
		center = np.array([l_h, l_w])//2
		assert (np.array((max_y, max_x)) == center).any(), 'center not corresponding'
		'''
		window = self.cos_window(self.ori_label.shape)
		self.ori_label = self.ori_label*window

		## EM to predict 2d Gaussian label
		return ori_label.copy()
		
	def track(self, x:'BGR image') -> 'response map':
		
		with torch.no_grad():
			xfs = self.backbone(x)
		xf = torch.cat(xfs, dim=1)
		xf = xf[:,self.select_idx,:,:]
		#xf.shape is a subclass of tuple
		#To change size, make it to numpy array
		#
		xf_h, xf_w = xf.shape[-2:]
		xf_size = np.array([xf_h, xf_w])
		larger_xf = F.interpolate(xf, tuple(xf_size+self.pad), mode='bilinear', align_corners=False)
		smaller_xf = F.interpolate(xf, tuple(xf_size-self.pad), mode='bilinear', align_corners=False)
		larger_xf_crop = larger_xf[:,:,self.margin_pad:xf_size[0]+self.margin_pad,self.margin_pad:xf_size[1]+self.margin_pad]
		smaller_xf_pad = F.pad(smaller_xf,(self.margin_pad,self.margin_pad,self.margin_pad,self.margin_pad), 'constant', 0)

		self.tri_xfs = torch.cat((larger_xf_crop, xf, smaller_xf_pad), dim=0)

		response = F.conv2d(self.tri_xfs, self.nor_zf)
		self.score = response

		pad_response = F.pad(response, self.p2d, "constant", 0)
		#assert tri_xfs.shape[-2:] == pad_response.shape[-2:]
		pad_response = pad_response/torch.max(pad_response)

		return pad_response

	def update(self, idx, displacement):
		# displacement is the deviation between adjacent frames
		score = self.score[idx].squeeze()
		sample = TensorList([self.tri_xfs[idx].unsqueeze(0)])
		dis_label = TensorList([self.translation(self.ori_label, displacement)])
		self.conv_problem = ConvProblem(sample, dis_label)
		self.filter_optimizer = ConjugateGradient(self.conv_problem, TensorList([self.nor_zf]), debug=0, plotting=0, cg_eps = 0.1)
		self.filter_optimizer.run(cfg.TEST.CG_ITER)
		new_filter = self.filter_optimizer.x
		#This mix up may optimize filter with occlusion object as in bird
		hyd_filter = (1-cfg.TEST.ONLINE_INFLUENCE)*self.nor_zf + cfg.TEST.ONLINE_INFLUENCE*torch.cat(new_filter)
		
		#new_score = F.conv2d(sample, hyd_filter).squeeze()
		self.nor_zf = hyd_filter
		
		if self.visdom is not None:
			loss = self.loss_function(dis_label, score)
			self.losses = torch.cat((self.losses, loss))
			new_loss = self.loss_function(dis_label, new_score)
			self.new_losses = torch.cat((self.new_losses, new_loss))

			self.visdom.register(score, 'heatmap', 2, 'Score Map')
			self.visdom.register(dis_label, 'heatmap', 2, 'Current Label')
			self.visdom.register(new_score, 'heatmap', 2, 'New Score')
			self.visdom.register(self.losses, 'lineplot', 3, 'Loss')
			self.visdom.register(self.new_losses, 'lineplot', 3, 'New Loss')

	#def optimize_boxes(self, idx, init_boxes):
	#	sample = self.tri_xfs[in]
	def update_net(self, idx, displacement):
		# displacement is the deviation between adjacent frames
		## define filter
		# Initialize filter
		filter_init_method = cfg.TEST.filter_init_method
		shape = self.nor_zf.shape
		self.filter = TensorList(
			[self.nor_zf.new_ones(shape[0], shape[1], shape[2], shape[3]),
			self.nor_zf.new_ones(shape[0], shape[1], shape[2], shape[3])])
		if filter_init_method == 'ones':
			pass
		elif filter_init_method == 'randn':
			for f in self.filter:
				f.normal_(0, 1/f.numel())
		else:
			raise ValueError('Unknown "filter_init_method"')
		## define activation
		# Activation function after the projection matrix (phi_1 in the paper)
		projection_activation = cfg.TEST.projection_activation
		if isinstance(projection_activation, tuple):
			projection_activation, act_param = projection_activation

		if projection_activation == 'none':
			self.projection_activation = lambda x: x
		elif projection_activation == 'relu':
			self.projection_activation = torch.nn.ReLU(inplace=True)
		elif projection_activation == 'elu':
			self.projection_activation = torch.nn.ELU(inplace=True)
		elif projection_activation == 'mlu':
			self.projection_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
		else:
			raise ValueError('Unknown activation')
		# Activation function after the output scores (phi_2 in the paper)
		response_activation = cfg.TEST.response_activation
		if isinstance(response_activation, tuple):
			response_activation, act_param = response_activation

		if response_activation == 'none':
			self.response_activation = lambda x: x
		elif response_activation == 'relu':
			self.response_activation = torch.nn.ReLU(inplace=True)
		elif response_activation == 'elu':
			self.response_activation = torch.nn.ELU(inplace=True)
		elif response_activation == 'mlu':
			self.response_activation = lambda x: F.elu(F.leaky_relu(x, 1 / act_param), act_param)
		else:
			raise ValueError('Unknown activation')
		## define 
		score = self.score[idx].squeeze()
		sample = self.tri_xfs[idx].unsqueeze(0)
		dis_label = self.translation(self.ori_label, displacement)
		self.conv_problem = FactorizedConvProblem_v2(sample, dis_label, self.nor_zf, 
														self.projection_activation, 
														self.response_activation)
		#self.conv_problem = ConvProblem(sample, dis_label)
		self.filter_optimizer = ConjugateGradient(self.conv_problem, self.filter, debug=0, plotting=0)
		self.filter_optimizer.run(cfg.TEST.CG_ITER)
		new_filter = self.filter_optimizer.x[1]*(self.filter_optimizer.x[0]*self.nor_zf)
		#This mix up may optimize filter with occlusion object as in bird
		hyd_filter = (1-cfg.TEST.ONLINE_INFLUENCE)*self.nor_zf + cfg.TEST.ONLINE_INFLUENCE*new_filter
		
		new_score = F.conv2d(sample, hyd_filter).squeeze()
		self.nor_zf = hyd_filter
		if self.visdom is not None:
			loss = self.loss_function(dis_label, score)
			self.losses = torch.cat((self.losses, loss))
			new_loss = self.loss_function(dis_label, new_score)
			self.new_losses = torch.cat((self.new_losses, new_loss))

			self.visdom.register(score, 'heatmap', 2, 'Score Map')
			self.visdom.register(dis_label, 'heatmap', 2, 'Current Label')
			self.visdom.register(new_score, 'heatmap', 2, 'New Score')
			self.visdom.register(self.losses, 'lineplot', 3, 'Loss')
			self.visdom.register(self.new_losses, 'lineplot', 3, 'New Loss')

	def translation(self, label:'np.ndarray', displacement):
		''''translate label from center'
		'''
		l_h, l_w = label.shape
		if isinstance(label, torch.Tensor):
			ori_label = label.cpu().numpy()
		moving_matrix = np.float64(np.array([[1,0,displacement[0]], [0,1,displacement[1]]]))
		dis_label = cv2.warpAffine(ori_label, moving_matrix, (l_w,l_h))
		return torch.from_numpy(dis_label).cuda()

	def loss_function(self, label, pred):
		loss = F.l1_loss(label, pred)
		#loss = F.mse_loss(label, pred)
		return loss.detach().cpu().view(-1)
