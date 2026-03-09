import torch
from detectron2.structures import Boxes, Instances

def semantic_inference(mask_cls, mask_pred, test_perfect_masks, test_with_void, test_with_fc_clip, test_metadata, object_mask_threshold, sim):
    # Try to discard masks
    # Pixels no label at all. Any mask that is void multiply the probabilities by 10^-3
    scores, labels = mask_cls.max(-1) # For each pixel, get the class with the highest score


    num_classes = len(test_metadata.stuff_classes)
    keep = (labels != num_classes) & (scores > object_mask_threshold)
    
    # Create a multiplier: 1 for good masks, 1e-1 for discarded masks
    discard_multiplier = keep.float() + (~keep).float() * 0.1  
    #mask_cls = mask_cls[keep] * discard_multiplier

    if not test_perfect_masks:
        if test_with_void or test_with_fc_clip:
            mask_cls = mask_cls[..., :-1]
            scores, labels = mask_cls.max(-1)
            mask_cls = mask_cls * sim[labels]
        mask_pred = mask_pred.sigmoid()

    semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
    return semseg

def panoptic_inference(mask_cls, mask_pred, test_perfect_masks, test_metadata, overlap_threshold, object_mask_threshold):
    scores, labels = mask_cls.max(-1) # For each pixel, get the class with the highest score

    if not test_perfect_masks:
        mask_pred = mask_pred.sigmoid()

    num_classes = len(test_metadata.stuff_classes)
    keep = labels.ne(num_classes) & (scores > object_mask_threshold) # Thresholding I guess. First part removes background I think
    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_masks = mask_pred[keep]

    cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks # Each pixel in the mask has the value of the score. Think that masks are 0,1

    h, w = cur_masks.shape[-2:]
    panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
    segments_info = []

    current_segment_id = 0

    if cur_masks.shape[0] == 0:
        # We didn't detect any mask :(
        return panoptic_seg, segments_info
    else:
        # take argmax
        cur_mask_ids = cur_prob_masks.argmax(0) # Gets the max score for the pixel. Actually the max index. Uses argmax
        stuff_memory_list = {}
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = pred_class in test_metadata.thing_dataset_id_to_contiguous_id.values()
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5) # Takes pixels of mask but only ones we are sure of 

            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < overlap_threshold: # The mask is covered by another
                    continue

                # merge stuff regions
                if not isthing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        continue
                    else:
                        stuff_memory_list[int(pred_class)] = current_segment_id + 1

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    }
                )

        return panoptic_seg, segments_info

def instance_inference(mask_cls, mask_pred, test_topk_per_image,
    test_with_void, test_with_fc_clip, panoptic_on, test_metadata, device):
    # mask_pred is already processed to have the same shape as original input
    image_size = mask_pred.shape[-2:]

    # [Q, K]
    scores = mask_cls.to(device)

    if test_with_void or test_with_fc_clip:
        scores = scores[:, :-1]

    # if this is panoptic segmentation
    if panoptic_on:
        num_classes = len(test_metadata.stuff_classes)
    else:
        num_classes = len(test_metadata.thing_classes)
    labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(mask_pred.shape[0], 1).flatten(0, 1)
    # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
    scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
    topk_indices.to(device)
    scores_per_image.to(device)
    labels_per_image = labels[topk_indices]

    topk_indices = topk_indices // num_classes
    # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
    mask_pred = mask_pred[topk_indices].to(device)

    # if this is panoptic segmentation, we only keep the "thing" classes
    if panoptic_on:
        keep = torch.zeros_like(labels_per_image).bool().to(device)
        for i, lab in enumerate(labels_per_image):
            keep[i] = lab in test_metadata.thing_dataset_id_to_contiguous_id.values()

        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]
        mask_pred = mask_pred[keep]

    result = Instances(image_size)
    result.pred_masks = (mask_pred > 0).float()
    result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

    # calculate average mask prob
    mask_scores_per_image = (mask_pred.flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
    result.scores = scores_per_image * mask_scores_per_image
    result.pred_classes = labels_per_image
    return result