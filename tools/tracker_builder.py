
from configs.default import cfg
from tracker.tssiamfc_tracker import TSSiamFCTracker

TRACKS = {
		  'TSSiamFC': TSSiamFCTracker,
          #'SiamRPNTracker': SiamRPNTracker,
          #'SiamMaskTracker': SiamMaskTracker,
          #'SiamRPNLTTracker': SiamRPNLTTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TEST.TYPE](model)