from typing import Tuple
import torch
import torch.nn.functional as F


def sem_seg_postprocess(
    result: torch.Tensor,
    img_size: Tuple[int, int],
    output_size: torch.Tensor,
):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    output_width = output_size[0].item()
    output_height = output_size[1].item()
    if img_size[0] == img_size[1]:
        max_side_length = max(output_height, output_width)
        result = F.interpolate(
            result.unsqueeze(0),
            size=(max_side_length, max_side_length),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        result = result[:, :output_height, :output_width]
    else:
        result = F.interpolate(
            result.unsqueeze(0),
            size=(output_height, output_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return result


@torch.jit.script
def pairwise_iou(masks1: torch.Tensor, masks2: torch.Tensor):
    masks1 = masks1.flatten(1)
    masks2 = masks2.flatten(1)

    intersection = torch.einsum("nc,mc->nm", masks1, masks2)
    union = masks1.sum(-1)[:, None] + masks2.sum(-1)[None, :]
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)

    return iou


@torch.jit.script
def pairwise_inner(masks1: torch.Tensor, masks2: torch.Tensor):
    masks1 = masks1.flatten(1)
    masks2 = masks2.flatten(1)
    
    inter = torch.einsum("nc,mc->nm", masks1, masks2)
    inner = (inter + 1e-6) / (masks2.sum(dim=1) + 1e-6)

    return inner


def nms(scores, iou_matrix, iou_threshold=0.7):
    # 按分数从高到低排序mask的索引
    _, order = scores.sort(descending=True)

    keep = []  # 保存NMS后的mask索引
    while order.numel() > 0:
        i = order[0]  # 当前最高分数的mask索引
        keep.append(i)  # 将当前最高分数的mask索引加入到keep列表

        if order.numel() == 1:  # 如果只剩下一个元素，则直接保留
            break

        # 计算当前mask与其他所有mask的IOU
        current_iou = iou_matrix[i, order[1:]]

        # 筛选出与当前mask IOU小于阈值的mask，它们不会被抑制
        remain_inds = torch.nonzero(current_iou < iou_threshold).squeeze(dim=1)

        # 更新order，只保留那些没有被当前mask抑制的mask的索引
        order = order[remain_inds + 1]  # 加1因为iou_matrix中排除了自己

    # 创建一个新的数组来存储NMS后的mask
    keep = torch.as_tensor(keep, dtype=torch.int64, device=scores.device)
    return keep


def inner_nms(scores, iou_matrix, ratio_matrix, iou_threshold=0.7, ratio_threshold=0.9):
    # 按分数从高到低排序mask的索引
    _, order = scores.sort(descending=True)

    keep = []  # 保存NMS后的mask索引
    while order.numel() > 0:
        i = order[0]  # 当前最高分数的mask索引
        keep.append(i)  # 将当前最高分数的mask索引加入到keep列表

        if order.numel() == 1:  # 如果只剩下一个元素，则直接保留
            break

        # 计算当前mask与其他所有mask的IOU
        current_iou = iou_matrix[i, order[1:]]
        current_ratio = ratio_matrix[i, order[1:]]

        # 筛选出与当前mask IOU小于阈值的mask，它们不会被抑制
        remain_inds = torch.nonzero((current_iou < iou_threshold) & \
            (current_ratio < ratio_threshold)).squeeze(dim=1)

        # 更新order，只保留那些没有被当前mask抑制的mask的索引
        order = order[remain_inds + 1]  # 加1因为iou_matrix中排除了自己

    # 创建一个新的数组来存储NMS后的mask
    keep = torch.as_tensor(keep, dtype=torch.int64, device=scores.device)
    return keep


def mask_nms(
    masks: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.75,
    inner_threshold: float = 0.9,
    nms_type: str = "nms",
    downsample: float = 1.0,
):
    """
    Performs non-maximum suppression (NMS) on the masks according to their intersection-over-union (IoU)
    overlap, independently for each instance. Masks are expected to be in ``(N, H, W)`` format, where N is
    the number of instances.

    Args:
        masks (Tensor): A tensor of shape ``(N, H, W)``, representing N masks of height H and width W.
        scores (Tensor): A tensor of shape ``(N,)`` representing the score of each mask.
        iou_threshold (float): A float representing the IoU threshold for deciding whether boxes overlap too
            much with respect to each other.

    Returns:
        Tensor: A tensor of shape ``(N,)`` representing the indices of the elements that have been kept by NMS.
    """
    # downsample mask
    if downsample < 1.0:
        masks = F.interpolate(
            masks.unsqueeze(0), scale_factor=downsample, mode="bilinear"
        ).squeeze(0)

    # flatten all masks
    masks = masks.reshape(masks.shape[0], -1)
    masks = masks.sigmoid().ge(0.5).float()

    # nms
    if nms_type == "nms":
        iou_matrix = pairwise_iou(masks, masks)
        keep = nms(scores, iou_matrix, iou_threshold)
    else:
        iou_matrix = pairwise_iou(masks, masks)
        inner_matrix = pairwise_inner(masks, masks)
        keep = inner_nms(scores, iou_matrix, inner_matrix, iou_threshold, inner_threshold)

    return keep


def batched_mask_nms(
    masks: torch.Tensor,
    scores: torch.Tensor,
    category_idxs: torch.Tensor,
    iou_threshold: float = 0.75,
    downsample: float = 1.0,
):
    """
    Performs batched non-maximum suppression (NMS) on the masks according to their intersection-over-union (IoU)
    overlap, independently for each category. Masks are expected to be in ``(N, H, W)`` format, where N is
    the number of instances.

    Args:
        masks (Tensor): A tensor of shape ``(N, H, W)``, representing N masks of height H and width W.
        scores (Tensor): A tensor of shape ``(N,)`` representing the score of each mask.
        category_idxs (Tensor): A tensor of shape ``(N,)`` representing the category index for each mask.
        iou_threshold (float): A float representing the IoU threshold for deciding whether boxes overlap too
            much with respect to each other.

    Returns:
        Tensor: A tensor of shape ``(N,)`` representing the indices of the elements that have been kept by NMS.
    """

    if masks.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=masks.device)

    # downsample mask
    if downsample < 1.0:
        masks = F.interpolate(
            masks.unsqueeze(0), scale_factor=downsample, mode="bilinear"
        ).squeeze(0)

    # Flatten masks and threshold
    masks_flat = masks.reshape(masks.shape[0], -1)
    masks_flat = masks_flat.sigmoid().ge(0.5).float()

    # Initialize tensor to keep track of the indices to keep
    keep_indices = torch.empty((0,), dtype=torch.int64, device=masks.device)

    # Process each category separately
    for category in torch.unique(category_idxs):
        # Filter masks and scores for the current category
        category_mask = category_idxs == category
        masks_category = masks_flat[category_mask]
        scores_category = scores[category_mask]

        # Compute pairwise IoU for masks in the current category
        iou = pairwise_iou(masks_category, masks_category)

        # Discard overlaps
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=0)
        category_keep = (iou_max <= iou_threshold).nonzero(as_tuple=False).squeeze(1)

        # Keep top scoring masks within this category
        if category_keep.numel() > 0:
            scores_keep = scores_category[category_keep]
            _, idx = scores_keep.sort(0, descending=True)
            category_keep = category_keep[idx]

        # Add indices (adjusted to original indexing) to keep_indices
        keep_indices = torch.cat(
            (
                keep_indices,
                torch.nonzero(category_mask, as_tuple=False).squeeze(1)[category_keep],
            )
        )

    return keep_indices


def get_classification_logits_fcclip(
    x, text_classifier, logit_scale=None, num_templates=None
):
    """
    x in shape of [B, *, C]
    text_classifier in shape of [num_classes, C]
    logit_scale is a learnable scalar https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L201
    return: [B, *, num_classes]
    """

    # Normalize feature vectors
    x = F.normalize(x, dim=-1)

    # Compute initial logits
    pred_logits = x @ text_classifier.transpose(-2, -1)  # Shape: B, *, N + 1

    # Efficiently compute the max ensemble (used in OpenSeg/ODISE)
    max_logits = []
    cur_idx = 0
    for num_t in num_templates:
        max_logits.append(pred_logits[:, :, cur_idx : cur_idx + num_t].amax(dim=-1))
        cur_idx += num_t
    final_pred_logits = torch.stack(max_logits, dim=-1)

    # Apply logit scale
    if logit_scale is not None:
        logit_scale = torch.clamp(logit_scale.exp(), max=100)
        final_pred_logits *= logit_scale

    return final_pred_logits
