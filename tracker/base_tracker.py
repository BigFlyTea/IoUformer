# Copyright (c) SenseTime. All Rights Reserved.

import cv2
import numpy as np
import torch

from config import cfg


class BaseTracker(object):
    """ Base tracker of single objec tracking
    """
    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox(list): [x, y, width, height]
                        x, y need to be 0-based
        """
        raise NotImplementedError

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        raise NotImplementedError


class SiameseTracker(BaseTracker):
    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        if isinstance(original_sz, float):
            original_sz = [original_sz, original_sz]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c[0] + 0.5)
        context_xmax = context_xmin + sz[0] - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c[1] + 0.5)
        context_ymax = context_ymin + sz[1] - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        if cfg.CUDA and torch.cuda.is_available():
            im_patch = im_patch.cuda()
        return im_patch
    def get_subwindow_v2(self, image:'BGR image', location:'ndarray:[x,y,w,h]', model_sz:'input size', visualize=False)->'subwindow':
        """crop search window, if out of size pad with border pixel
        """
        size = np.array(location[2:4])
        position = np.array(location[0:2]) + size/2
        height, width = image.shape[0:2]


        x_index = (np.floor(position[0] + np.arange(1, size[0]+1) - size[0]/2)).astype(int)
        y_index = (np.floor(position[1] + np.arange(1, size[1]+1) - size[1]/2)).astype(int)

        #crop
        y_index = self.clamp(y_index, 0, height - 1)

        x_index = self.clamp(x_index, 0, width - 1)

        x_index, y_index = np.meshgrid(x_index, y_index)
        #ori_image = image
        img_patch = image[y_index, x_index, :]
        '''
        ##enlarge small object##
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel('./FSRCNN-small_x4.pb')
        sr.setModel('fsrcnn', 4)
        sr_img = sr.upsample(image)
        '''
        if isinstance(model_sz, (float, int)):
            model_sz = (model_sz, model_sz)

        img_patch = cv2.resize(img_patch, tuple(model_sz), interpolation=cv2.INTER_LINEAR)

        if False:

            image = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)
            print('show_image')
            cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('input_image', image)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()

        # image transforms for matconvnet vgg model
        # RGB
        # fixed-size 224x224
        # substract the mean RGB value 128

        #image transforms for matconvnet vgg model
        #image = torch.tensor(image.transpose(2,0,1), dtype=torch.float)- 128
        img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
        img_patch = img_patch.transpose(2,0,1)
        img_patch = img_patch[np.newaxis, :, :, :]
        img_patch = img_patch.astype(np.float32)
        img_patch = torch.from_numpy(img_patch) - 128
        if cfg.CUDA and torch.cuda.is_available():
            img_patch = img_patch.cuda()
        else:
            img_patch = img_patch.cpu()

        return img_patch

    def clamp(self, index, lower, upper):
        '''truncate coordinate to keep in image size,
           outside coordinate is truncated to border
        '''
        for idx in range(len(index)):
            if index[idx] > upper:
                index[idx] = upper
            if index[idx] < lower:
                index[idx] = lower
        return index
