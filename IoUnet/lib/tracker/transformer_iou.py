
import torch
from IoUnet.lib.tracker.basetracker import BaseTracker
from IoUnet.data.processing_utils import sample_target
# for debug
import cv2
import os
import numpy as np
from IoUnet.utils.merge import merge_template_search
from IoUnet.lib.models import build_iounet
from IoUnet.lib.tracker.stark_utils import Preprocessor
from IoUnet.utils.box_ops import clip_box
import IoUnet.data.processing_utils as prutils
from config import cfg
class Transformer_iou(BaseTracker):
    def __init__(self, params):
        super(Transformer_iou, self).__init__(params)
        network = build_iounet(params.cfg)
        print(self.params.checkpoint)
        #net_wight = torch.load(self.params.checkpoint)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu'), strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        self.debug = True
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

        
    def initialize(self, image, init_bb, hp):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, init_bb, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template)
        # save states
        self.state = init_bb
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = init_bb * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}
        # training setting
        #self.proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 16, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}
        self.proposal_params = {'min_iou': cfg.TEST.min_iou, 
                               'boxes_per_frame': cfg.TEST.boxes_per_frame, 
                               'sigma_factor': cfg.TEST.sigma_factor}
    def _generate_proposals(self, box):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        
        proposal_method = self.proposal_params.get('proposal_method', 'default')
        if proposal_method == 'default':
            proposals = torch.zeros((num_proposals, 4))
            gt_iou = torch.zeros(num_proposals)
            for i in range(num_proposals):
                proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=self.proposal_params['min_iou'],
                                                                 sigma_factor=self.proposal_params['sigma_factor'])
        elif proposal_method == 'gmm':
            proposals, _, _ = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                             num_samples=num_proposals)
            gt_iou = prutils.iou(box.view(1,4), proposals.view(-1,4))

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def track(self, image, pred_bb):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape
        self.frame_id += 1
        self.state = pred_bb
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        ## transform state in original space to crop space
        bb = torch.as_tensor(self.state.copy(), dtype=torch.float32)
        out_sz = torch.Tensor([self.params.search_size, self.params.search_size])
        out_center = out_sz/2
        bb_crop_wh = bb[2:4]*resize_factor

        bb_crop = torch.cat((out_center-0.5*bb_crop_wh, bb_crop_wh))
        proposals, iou_pre = self._generate_proposals(bb_crop)
        proposals = proposals.unsqueeze(0).cuda()
        #print(proposals.shape)
        
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = self.network.forward_backbone(search, template=False, proposals=proposals)
            # merge the template and the search
            feat_dict_list = [self.z_dict1, x_dict]
            seq_dict = merge_template_search(feat_dict_list)
            # run the transformer
            out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True)
        
        '''
        K=5
        topk_iou, topk_ind = torch.topk(out_dict['ious'],K, dim=0)
        
        if K==1:
            pred_bbs = proposals[:,topk_ind,:].squeeze()
        else:
            pred_bbs = proposals[:,topk_ind,:].squeeze().mean(dim=0)
        '''
        select_ind = (out_dict['ious']>cfg.TEST.IOU_THRESHOLD).squeeze()
        if (select_ind==False).all():
            return pred_bb
        else:
            pred_bbs = proposals[:,select_ind,:][0].mean(dim=0)
            new_wh = pred_bbs[2:4]/resize_factor
            ## the displacement is inaccurate?
            displacement = (pred_bbs[0:2]+0.5*pred_bbs[2:4]-out_center.cuda())/resize_factor
            cx_prev, cy_prev = bb[0]+0.5*bb[2], bb[1]+0.5*bb[3]
            #print(displacement)
            cx_new, cy_new = cx_prev+displacement[0], cy_prev+displacement[1]
            xy = torch.as_tensor((cx_new-0.5*new_wh[0], cy_new-0.5*new_wh[1])).cuda()
            
            new_state = torch.cat((xy, new_wh)).tolist()
            #print(bb)
            #print('new', new_state)
            self.state = clip_box(new_state, H, W, margin=10)
            return np.array(self.state)
            

        '''
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        '''
        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            print(save_path)
            cv2.imwrite(save_path, image_BGR)
            
        
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return Transformer_iou
