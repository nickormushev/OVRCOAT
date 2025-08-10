import torch
from torch.nn import functional as F

def batched_cosine_similarity_loss(A, B):
    A_normalized = F.normalize(A, p=2, dim=2) 
    B_normalized = F.normalize(B, p=2, dim=2)
    
    cosine_sim = torch.sum(A_normalized * B_normalized, dim=2)  # (B, m)
    
    mean_cosine_sim = torch.mean(cosine_sim, dim=1)  # (B,)
    
    loss = 1 - torch.mean(mean_cosine_sim)  # Scalar loss
    return loss

def linear_warmup(step, total_warmup_steps, final_value):
    if total_warmup_steps == 0:
        return final_value
    return min(final_value, final_value * step / total_warmup_steps)

def get_pooled_features(self, clip_features):
    pooled_features = []
    for K in range(1,3):
       pooled_features.append(F.avg_pool2d(clip_features, kernel_size=(K, K)))
    return torch.cat(pooled_features, dim=0)
    
def calculate_dist_loss(clip_feature, frozen_clip_feature, loss, iteration, dist_warmup_iters, weight_dict):
    # Reshape clip_feature and frozen_clip_feature to [batch_size, num_channels * height * width]
    reshaped_clip_feat = clip_feature.view(clip_feature.shape[0], clip_feature.shape[1], -1)
    reshaped_frozen_clip_feat = frozen_clip_feature.view(frozen_clip_feature.shape[0],
                                                        frozen_clip_feature.shape[1], -1)

    reshaped_clip_feat = F.normalize(reshaped_clip_feat, dim=2)  # normalise along spatial dims
    reshaped_frozen_clip_feat = F.normalize(reshaped_frozen_clip_feat, dim=2)

    gram_matrix_clip_feat = torch.bmm(reshaped_clip_feat, reshaped_clip_feat.transpose(1, 2))
    gram_matrix_frozen_clip_feat = torch.bmm(reshaped_frozen_clip_feat, reshaped_frozen_clip_feat.transpose(1, 2)) 

    dist_weight = linear_warmup(iteration, dist_warmup_iters, weight_dict["dist_loss"])

    if loss == "l2":
        return dist_weight * F.mse_loss(gram_matrix_clip_feat, gram_matrix_frozen_clip_feat)
    
    if loss == "smoothl1":
        return dist_weight * F.smooth_l1_loss(gram_matrix_clip_feat, gram_matrix_frozen_clip_feat)
    
    if loss == "maft":
        pooled_clip_feature = get_pooled_features(clip_feature)
        pooled_frozen_clip_feature = get_pooled_features(frozen_clip_feature)
        return dist_weight * F.smooth_l1_loss(pooled_clip_feature, pooled_frozen_clip_feature)
    
    return dist_weight * batched_cosine_similarity_loss(gram_matrix_clip_feat, gram_matrix_frozen_clip_feat)


def calculate_ce_loss(out_vocab_cls_results, gt_labels):
    batch_size, num_masks, num_classes = out_vocab_cls_results.shape

    # Reshape out_vocab_cls_results to [batch_size * num_objects, num_classes]
    out_vocab_cls_results = out_vocab_cls_results.reshape(batch_size * num_masks, num_classes)

    # Reshape gt_labels to [batch_size * num_objects]
    gt_labels = gt_labels.reshape(batch_size * num_masks)
    return F.cross_entropy(out_vocab_cls_results, gt_labels, reduction='mean')
