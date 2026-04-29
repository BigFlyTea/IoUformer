import torch


def merge_template_search(inp_list, return_search=False, return_template=False):
    """NOTICE: search region related features must be in the last place"""
    ## can not repeat in this, cause the not aligned features
    '''
    template_batch = inp_list[0]['mask'].shape[0]
    search_batch = inp_list[1]['mask'].size(0)
    rois_per_sample = int(search_batch/template_batch)
    ### repeat to expand template batch size ###
    template_feat = inp_list[0]['feat']
    template_feat_repeat = template_feat.repeat(1, rois_per_sample, 1)
    template_mask = inp_list[0]['mask']
    template_mask_repeat = template_mask.repeat(rois_per_sample, 1)
    
    print(template_mask_repeat.shape)
    print(inp_list[1]['mask'].shape)
    template_pos = inp_list[0]['pos']
    template_pos_repeat = template_pos.repeat(1, rois_per_sample, 1)

    inp_list[0]['feat'] = template_feat_repeat
    inp_list[0]['mask'] = template_mask_repeat
    inp_list[0]['pos'] = template_pos_repeat
    '''
    
    seq_dict = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
                "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
                "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}
    if return_search:
        x = inp_list[-1]
        seq_dict.update({"feat_x": x["feat"], "mask_x": x["mask"], "pos_x": x["pos"]})
    if return_template:
        z = inp_list[0]
        seq_dict.update({"feat_z": z["feat"], "mask_z": z["mask"], "pos_z": z["pos"]})
    
    return seq_dict


def get_qkv(inp_list):
    """The 1st element of the inp_list is about the template,
    the 2nd (the last) element is about the search region"""
    dict_x = inp_list[-1]
    dict_c = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
              "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
              "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}  # concatenated dict
    q = dict_x["feat"] + dict_x["pos"]
    k = dict_c["feat"] + dict_c["pos"]
    v = dict_c["feat"]
    key_padding_mask = dict_c["mask"]
    return q, k, v, key_padding_mask
