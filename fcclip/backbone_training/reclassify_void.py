import torch
import numpy as np
from torch.nn import functional as F

def reclassify_void_masks(num_classes, pred_clfs, clip_preds, clip_similarities, 
                            thing_dataset_id_to_contiguous_id, device='cuda',
                            give_things_priority = True, clip_treshold=0,
                            sim_threshold=26.5, softmax_temperature=5):

    pred_clfs_np = pred_clfs.cpu().detach().numpy()
    clip_preds_np = clip_preds.cpu().detach().numpy()
    new_mask_cls = pred_clfs_np

    for i in range(pred_clfs_np.shape[0]): 
        pred_category = np.argmax(pred_clfs_np[i])
        pred_is_background = pred_category == (num_classes - 1)
        clip_category = np.argmax(clip_preds_np[i])
        clip_prob = np.max(clip_preds_np[i])
        similarity = clip_similarities[i, clip_category]

        is_thing = clip_category in thing_dataset_id_to_contiguous_id.values()

        if pred_is_background and clip_prob >= clip_treshold and similarity > sim_threshold:
            new_mask_cls[i, 0:num_classes - 1] = F.softmax(clip_similarities[i]/softmax_temperature, dim=-1).cpu().detach().numpy()
            new_mask_cls[i, num_classes - 1] = 0
            new_mask_cls[i, clip_category] += 0.2 if is_thing and give_things_priority else 0

    return torch.tensor(new_mask_cls, device=device)