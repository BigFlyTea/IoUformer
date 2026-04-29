from IoUnet.parameter import TrackerParams
import os

from IoUnet.config.transformer_iou.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    
    # update default config from yaml file
    yaml_file = os.path.join('/media/zgluo/laboratory/pytorch/TSSiamFC_visdom_v0_IoUnet/IoUnet', 'parameter/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join('/media/zgluo/laboratory/pytorch/TSSiamFC_visdom_v0_IoUnet/IoUnet', "pretrained/model%04d.pth" %
                                     (cfg.TEST.EPOCH))

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
