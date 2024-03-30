import logging
from typing import Dict, List, Tuple

import torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom
from torch import nn
from torch.nn import functional as F

from ..backbone.clip import build_main_backbone
from ...utils import (
    mask_nms,
    sem_seg_postprocess,
)
from ..pixel_decoder import build_pixel_decoder
from ..transformer_decoder import build_transformer_decoder
from ..utils import MaskPooling

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class UniOVSeg_S1(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        pixel_decoder: nn.Module,
        mask_decoder: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        pts_per_side_test: int,
        input_size: Tuple[int],
        autolabel_type: str = "panoptic-point",
        sem_seg_postprocess_before_inference: bool,
    ):
        super().__init__()
        # architecture
        self.backbone = backbone
        self.sem_seg_head = nn.ModuleDict(
            {
                "pixel_decoder": pixel_decoder,
                "predictor": mask_decoder,
            }
        )
        self.mask_pooling = MaskPooling()

        # utils
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference

        # test settings
        autolabel_type = autolabel_type.split("-")
        self.autolabel_type = autolabel_type[0]
        self.prompt_type = autolabel_type[1]

        # image statistics
        self.register_buffer(
            name="pixel_mean",
            tensor=torch.Tensor(pixel_mean).view(-1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            name="pixel_std",
            tensor=torch.Tensor(pixel_std).view(-1, 1, 1),
            persistent=False,
        )

        # point prompts
        self.pts_per_side_test = pts_per_side_test
        self.input_size = input_size

    @classmethod
    def from_config(cls, cfg):
        backbone = build_main_backbone(cfg)
        pixel_decoder = build_pixel_decoder(cfg, backbone.output_shape())
        mask_decoder = build_transformer_decoder(cfg)
        return {
            "backbone": backbone,
            "pixel_decoder": pixel_decoder,
            "mask_decoder": mask_decoder,
            "size_divisibility": cfg.MODEL.OVSEG.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "pts_per_side_test": cfg.MODEL.OVSEG.TEST.PTS_PER_SIDE,
            "input_size": (cfg.INPUT.CROP_SIZE, cfg.INPUT.CROP_SIZE),
            "autolabel_type": cfg.MODEL.OVSEG.TEST.AUTOLABEL_TYPE,
            "sem_seg_postprocess_before_inference": cfg.MODEL.OVSEG.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def preprosses_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        sizes = torch.tensor(
            [
                [
                    x.get("width", self.input_size[0]) - 1,
                    x.get("height", self.input_size[1]) - 1,
                ]
                for x in batched_inputs
            ],
            device=self.device,
        )
        aspect_ratio = (self.input_size[0] - 1.0) / sizes.amax(dim=-1)
        aspect_ratio = aspect_ratio.unsqueeze(1)
        scaled_sizes = sizes * aspect_ratio
        sizes = sizes + 1
        return images, sizes, scaled_sizes

    @staticmethod
    def prepare_points(
        pts_per_side: Tuple[int, int], scaled_size: torch.Tensor, device: torch.device
    ):
        pts_side_x, pts_side_y = pts_per_side
        offset_x = 1 / (2 * pts_side_x)
        offset_y = 1 / (2 * pts_side_y)
        pts_x_side = torch.linspace(offset_x, 1 - offset_x, pts_side_x)
        pts_y_side = torch.linspace(offset_y, 1 - offset_y, pts_side_y)
        pts_x, pts_y = torch.meshgrid(pts_x_side, pts_y_side, indexing="xy")
        pts_grid = torch.stack([pts_x, pts_y], dim=-1).reshape(-1, 2)
        pts_grid = pts_grid.to(device)
        # scale to image size
        pts_grid = pts_grid.unsqueeze(0) * scaled_size.unsqueeze(1)
        pts_grid = torch.cat([pts_grid - 2.0, pts_grid + 2.0], dim=-1)
        return pts_grid

    def pts_test_forward(
        self,
        features: Dict,
        scaled_sizes: torch.Tensor,
    ):

        (
            mask_features,
            transformer_encoder_features,
            multi_scale_features,
        ) = self.sem_seg_head["pixel_decoder"].forward_features(features)

        # slice forward for memory-efficient inference
        points = self.prepare_points(self.pts_per_side_test, scaled_sizes, self.device)
        points = torch.split(points.squeeze(0), 100, dim=0)
        mask_pred_results, iou_pred_results = [], []
        for point in points:
            output = self.sem_seg_head["predictor"](
                multi_scale_features,
                mask_features,
                points=[point],
                boxes=None,
                points_multi=None,
            )
            mask_pred_results.append(output["pred_masks"].flatten(1, 2))
            iou_pred_results.append(output["pred_ious"].flatten(1, 2))
        del points, point, output
        mask_pred_results = torch.cat(mask_pred_results, dim=1)
        iou_pred_results = torch.cat(iou_pred_results, dim=1)

        return mask_pred_results, iou_pred_results

    def box_test_forward(
        self,
        batched_inputs: List,
        features: Dict,
    ):
        boxes = [x["instances"].gt_boxes.to(self.device) for x in batched_inputs]
        (
            mask_features,
            transformer_encoder_features,
            multi_scale_features,
        ) = self.sem_seg_head["pixel_decoder"].forward_features(features)
        outputs = self.sem_seg_head["predictor"](
            multi_scale_features,
            mask_features,
            points=None,
            boxes=boxes,
            points_multi=None,
        )
        mask_pred_results = outputs["pred_masks"].flatten(1, 2)  # N, QK, H, W
        iou_pred_results = outputs["pred_ious"].flatten(1, 2)  # N, QK

        return mask_pred_results, iou_pred_results

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        # prepare image
        images, output_sizes, scaled_sizes = self.preprosses_image(batched_inputs)
        # feature extraction via backbone
        features = self.backbone(images.tensor)
        # mask prediction
        if self.prompt_type == "point":
            mask_pred_results, iou_pred_results = self.pts_test_forward(
                features, scaled_sizes
            )
        elif self.prompt_type == "box":
            mask_pred_results, iou_pred_results = self.box_test_forward(
                batched_inputs, features
            )
        else:
            raise NotImplementedError(
                f"Visual prompt type {self.prompt_type} is not supported"
            )
        # post-process
        iou_pred_results = iou_pred_results.sigmoid()
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )
        processed_results = []
        for mask_pred_result, iou_pred_result, output_size in zip(
            mask_pred_results, iou_pred_results, output_sizes
        ):
            processed_results.append({})
            # drop low iou predictions
            keep = iou_pred_result.ge(0.65)
            iou_pred_result = iou_pred_result[keep]
            mask_pred_result = mask_pred_result[keep]
            # drop redundant mask via mask nms
            keep = mask_nms(
                masks=mask_pred_result,
                scores=iou_pred_result,
                iou_threshold=0.5,
                inner_threshold=0.7,
                nms_type="inner-nms",
                downsample=0.5,
            )
            iou_pred_result = iou_pred_result[keep]
            mask_pred_result = mask_pred_result[keep]
            # semseg post-processing
            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, self.input_size, output_size
            )
            processed_results[-1]["proposal"] = mask_pred_result
        torch.cuda.empty_cache()
        return processed_results
