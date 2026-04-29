"""
Basic STARK Model (Spatial-only).
"""
import torch
from torch import nn

from IoUnet.utils.misc import NestedTensor

from .backbone import build_backbone
from .transformer import build_transformer
from .head import build_box_head
from IoUnet.utils.box_ops import box_xyxy_to_cxcywh
from IoUnet.lib.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from config import cfg
class STARKS(nn.Module):
	""" This is the base class for Transformer Tracking """
	def __init__(self, backbone, transformer, box_head, num_queries,
				 aux_loss=False, head_type="CORNER"):
		""" Initializes the model.
		Parameters:
			backbone: torch module of the backbone to be used. See backbone.py
			transformer: torch module of the transformer architecture. See transformer.py
			num_queries: number of object queries.
			aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
		"""
		super().__init__()
		self.backbone = backbone
		self.transformer = transformer
		self.box_head = box_head
		self.num_queries = num_queries
		hidden_dim = transformer.d_model
		self.query_embed = nn.Embedding(num_queries, hidden_dim)  # object queries
		self.bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer
		
		self.prroi_pool = PrRoIPool2D(8, 8, 1/16)

		self.aux_loss = aux_loss
		self.head_type = head_type
		if head_type == "CORNER":
			self.feat_sz_s = int(box_head.feat_sz)
			self.feat_len_s = int(box_head.feat_sz ** 2)

	def forward(self, img=None, seq_dict=None, mode="backbone", run_box_head=True, run_cls_head=False, template=True, proposals=None):
		if mode == "backbone":
			return self.forward_backbone(img, template, proposals)
		elif mode == "transformer":
			return self.forward_transformer(seq_dict, run_box_head=run_box_head, run_cls_head=run_cls_head)
		else:
			raise ValueError

	def forward_backbone(self, input: NestedTensor, template=True, proposals=None):
		"""The input type is NestedTensor, which consists of:
			   - tensor: batched images, of shape [batch_size x 3 x H x W]
			   - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
		"""
		assert isinstance(input, NestedTensor)
		# Forward the backbone
		output_back, pos = self.backbone(input)  # features & masks, position embedding for the search
		# Adjust the shapes
		return self.adjust(output_back, pos, template, proposals)

	def forward_transformer(self, seq_dict, run_box_head=True, run_cls_head=False):
		if self.aux_loss:
			raise ValueError("Deep supervision is not supported.")
		# Forward the transformer encoder and decoder
		output_embed, enc_mem = self.transformer(seq_dict["feat"], seq_dict["mask"], self.query_embed.weight,
												 seq_dict["pos"], return_encoder_output=True)
		# Forward the corner head
		out, outputs_coord = self.forward_box_head(output_embed, enc_mem)
		return out, outputs_coord, output_embed

	def forward_box_head(self, hs, memory):
		"""
		hs: output embeddings (1, B, N, C)
		memory: encoder embeddings (HW1+HW2, B, C)"""
		self.head_type = "IOU"
		if self.head_type == "IOU":
			# adjust shape
			enc_opt = memory[-8**2:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
			dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
			
			att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
			opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
			bs, Nq, C, HW = opt.size() #Nq=1
			#print(opt.size())
			opt_feat = opt.view(-1, C, 8, 8)
			
			ious = self.box_head(opt_feat)
			out = {'ious':ious}
			return out, ious
		if self.head_type == "CORNER":
			# adjust shape
			enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
			dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
			att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
			opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
			bs, Nq, C, HW = opt.size()
			opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
			# run the corner head
			outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
			outputs_coord_new = outputs_coord.view(bs, Nq, 4)
			out = {'pred_boxes': outputs_coord_new}
			return out, outputs_coord_new
		elif self.head_type == "MLP":
			# Forward the class and box head
			outputs_coord = self.box_head(hs).sigmoid()
			out = {'pred_boxes': outputs_coord[-1]}
			if self.aux_loss:
				out['aux_outputs'] = self._set_aux_loss(outputs_coord)
			return out, outputs_coord

	def adjust(self, output_back: list, pos_embed: list, template:'Flag', proposals:'rois'):
		"""
		"""
		src_feat, mask = output_back[-1].decompose()
		num_proposals_per_batch = int(cfg.TEST.boxes_per_frame)
		assert mask is not None
		# reduce channel
		feat = self.bottleneck(src_feat)  # (B, C, H, W)
		B,C,H,W = feat.shape
		### roi pool in deep feature and mask?
		if not template and proposals is not None:
		   # input proposals2 is in format xywh, convert it to x0y0x1y1 format
		   proposals_xyxy = torch.cat((proposals[:, :, 0:2], proposals[:, :, 0:2] + proposals[:, :, 2:4]), dim=2)
		   # Add batch_index to rois
		   batch_size = proposals.size()[0]
		   num_proposals_per_batch = proposals.shape[1]
		   batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(feat.device)
		   
		   roi2 = torch.cat((batch_index.reshape(batch_size, -1, 1).expand(-1, num_proposals_per_batch, -1),
						  proposals_xyxy), dim=2)
		   roi2 = roi2.reshape(-1, 5)
		   search_roi_feats =  self.prroi_pool(feat, roi2) #(B*num, C, H,W)
		   search_roi_pos = self.prroi_pool(pos_embed[-1], roi2)
		   mask = mask.to(torch.float).unsqueeze(1)
		   search_roi_mask = self.prroi_pool(mask, roi2)
		   search_roi_mask = search_roi_mask.to(torch.bool)
		   
		   search_rois_vec = search_roi_feats.flatten(2).permute(2,0,1)#(HW, B*num, C)
		   search_embed_vec = search_roi_pos.flatten(2).permute(2,0,1)
		   search_mask_vec = search_roi_mask.squeeze().flatten(1)

		   return {"feat": search_rois_vec, "mask":search_mask_vec, "pos":search_embed_vec}
		
		if template:
			feat = feat.repeat(1, num_proposals_per_batch, 1, 1) #(B, num*C, H, W)
			feat = feat.reshape(-1, C, H, W)
			mask = mask.unsqueeze(1).repeat(1, num_proposals_per_batch, 1, 1) #(B, num*C, H, W)
			mask = mask.reshape(-1, H, W)
			pos = pos_embed[-1].repeat(1, num_proposals_per_batch, 1, 1) #(B, num*C, H, W)
			pos = pos.reshape(-1, C, H, W)
			# adjust shapes
			feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
			pos_embed_vec = pos.flatten(2).permute(2, 0, 1)  # HWxBxC
			mask_vec = mask.flatten(1)  # BxHW
			return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

	@torch.jit.unused
	def _set_aux_loss(self, outputs_coord):
		# this is a workaround to make torchscript happy, as torchscript
		# doesn't support dictionary with non-homogeneous values, such
		# as a dict having both a Tensor and a list.
		return [{'pred_boxes': b}
				for b in outputs_coord[:-1]]


def build_iounet(cfg):
	backbone = build_backbone(cfg)  # backbone and positional encoding are built together
	transformer = build_transformer(cfg)
	box_head = build_box_head(cfg)
	model = STARKS(
		backbone,
		transformer,
		box_head,
		num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
		aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
		head_type=cfg.MODEL.HEAD_TYPE
	)

	return model
