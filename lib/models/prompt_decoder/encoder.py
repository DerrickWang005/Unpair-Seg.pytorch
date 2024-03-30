from typing import List, Tuple
import torch
import torch.nn as nn

from detectron2.config import configurable
from detectron2.utils.registry import Registry
from ..transformer_decoder.position_encoding import (
    PositionEmbeddingSine,
    PositionEmbeddingRandom,
)


PROMPT_ENCODER_REGISTRY = Registry("PROMPT_ENCODER")
PROMPT_ENCODER_REGISTRY.__doc__ = """
Registry for prompt encoder in Uni-OVSeg.
"""


def build_prompt_encoder(cfg):
    """
    Build a prompt encoder from `cfg.MODEL.OVSEG.PROMPT_ENCODER_NAME`.
    """
    name = cfg.MODEL.OVSEG.PROMPT_ENCODER_NAME
    model = PROMPT_ENCODER_REGISTRY.get(name)(cfg)
    return model


@PROMPT_ENCODER_REGISTRY.register()
class PromptEncoder(nn.Module):
    @configurable
    def __init__(self, embed_dim: int, image_size: List[int], num_masks: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        # position embedding
        # self.pos_emb = PositionEmbeddingSine(embed_dim // 2, normalize=True)
        self.pos_emb = PositionEmbeddingRandom(embed_dim // 2)
        # corner embedding: left top, right bottom
        self.corner_emb = nn.Parameter(torch.randn(1, 2, embed_dim))
        nn.init.normal_(self.corner_emb)
        # type embedding: point, box
        self.point_emb = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.point_emb)
        self.box_emb = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.box_emb)
        # attribute embedding: positive, negative
        self.attr_emb = nn.Embedding(2, embed_dim)
        nn.init.normal_(self.attr_emb.weight)
        # mask embedding: num_masks mask proposals
        self.mask_emb = nn.Parameter(torch.randn(1, num_masks + 1, embed_dim))
        nn.init.normal_(self.mask_emb)

    def freeze_everything(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        ret = {}
        ret["embed_dim"] = cfg.MODEL.OVSEG.EMBED_DIM
        ret["image_size"] = [cfg.INPUT.CROP_SIZE, cfg.INPUT.CROP_SIZE]
        ret["num_masks"] = cfg.MODEL.OVSEG.NUM_MASKS
        return ret

    def _embed_point(self, box: torch.Tensor, feat: torch.Tensor):
        N = len(box)
        if hasattr(box, "tensor"):
            box = box.tensor + 0.5
        box_embed = self.pos_emb.forward_with_coords(
            box.reshape(N, 2, 2), self.image_size
        )
        attr_embed = self.attr_emb(torch.ones_like(box_embed[:, :, 0]).long())
        corner_embed = self.corner_emb.clone()
        point_embed = self.point_emb.clone()
        content_embed = feat.unsqueeze(1)
        task_embed = box_embed + corner_embed + point_embed + content_embed + attr_embed

        output_embed = self.mask_emb.repeat(N, 1, 1)  # 1 x num_masks x C
        task_embed = torch.cat(
            [task_embed, output_embed], dim=1
        )  # N x (2 + num_masks) x C
        return task_embed, task_embed

    def _embed_point2(
        self, box: torch.Tensor, indicator: torch.Tensor, feat: torch.Tensor
    ):
        N, P, _ = box.shape
        box = box.reshape(N * P, 2, 2)
        box_embed = self.pos_emb.forward_with_coords(box, self.image_size)
        box_embed = box_embed.reshape(N, P, 2, -1)
        attr_embed = self.attr_emb(indicator.long()).unsqueeze(2)
        corner_embed = self.corner_emb.clone().unsqueeze(0)
        point_embed = self.point_emb.clone().unsqueeze(0)
        content_embed = feat.unsqueeze(2)
        task_embed = box_embed + corner_embed + point_embed + content_embed + attr_embed
        task_embed = task_embed.reshape(N, P * 2, -1)

        # output_embed = self.mask_emb_single.repeat(N, 1, 1)  # N x 1 x C
        output_embed = self.mask_emb.repeat(N, 1, 1)[:, :1]  # N x 1 x C
        task_embed = torch.cat([task_embed, output_embed], dim=1)  # N x (2P + 1) x C
        return task_embed, task_embed

    def _embed_box(self, box: torch.Tensor, feat: torch.Tensor):
        N = len(box)
        if hasattr(box, "tensor"):
            box = box.tensor + 0.5
        box_embed = self.pos_emb.forward_with_coords(
            box.reshape(N, 2, 2), self.image_size
        )
        corner_embed = self.corner_emb.clone()
        point_embed = self.box_emb.clone()
        content_embed = feat.unsqueeze(1)
        task_embed = box_embed + corner_embed + point_embed + content_embed

        output_embed = self.mask_emb.repeat(N, 1, 1)  # 1 x num_masks x C
        task_embed = torch.cat(
            [task_embed, output_embed], dim=1
        )  # N x (2 + num_masks) x C
        return task_embed, task_embed

    def forward(
        self,
        points: List[torch.Tensor],
        boxes: List[torch.Tensor],
        points_multi: Tuple[List],
        feats_centers: torch.Tensor,
    ):
        """This is a forward function of embedding multi-type prompts

        Args:
            points (List[torch.Tensor]): A batch of point coordinates. Each one has a shape of [Q, 4].
            feats_centers (torch.Tensor): A batch of center features. It has a shape of [B, Q, C]
        Return:
            List[torch.Tensor]: Prompt embedding has a shape of [B, Q, K, C]
        """
        # embed input prompt into a embedding space
        task_emb, pos_emb = [], []
        if points is not None:
            for pts, feat in zip(points, feats_centers):
                task, pos = self._embed_point(pts, feat)
                task_emb.append(task)
                pos_emb.append(pos)
        if boxes is not None:
            for pts, feat in zip(boxes, feats_centers):
                task, pos = self._embed_box(pts, feat)
                task_emb.append(task)
                pos_emb.append(pos)
        if points_multi is not None:
            for pts, ind, feat in zip(points_multi[0], points_multi[1], feats_centers):
                task, pos = self._embed_point2(pts, ind, feat)
                task_emb.append(task)
                pos_emb.append(pos)

        task_emb = torch.stack(task_emb, dim=0)  # [B, Q, K, C]
        pos_emb = torch.stack(pos_emb, dim=0)  # [B, Q, K, C]

        return task_emb, pos_emb


@PROMPT_ENCODER_REGISTRY.register()
class PromptEncoder2(nn.Module):
    @configurable
    def __init__(self, embed_dim: int, image_size: List[int], num_masks: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        # position embedding
        self.pos_emb = PositionEmbeddingSine(embed_dim // 2, normalize=True)
        # corner embedding: left top, right bottom
        self.corner_emb = nn.Parameter(torch.randn(1, 2, embed_dim))
        nn.init.normal_(self.corner_emb)
        # type embedding: point, box
        self.point_emb = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.point_emb)
        self.box_emb = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.box_emb)
        # attribute embedding: positive, negative
        self.attr_emb = nn.Embedding(2, embed_dim)
        nn.init.normal_(self.attr_emb.weight)
        # mask embedding: num_masks mask proposals
        self.mask_emb = nn.Parameter(torch.randn(1, num_masks + 2, embed_dim))
        nn.init.normal_(self.mask_emb)

    def freeze_everything(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        ret = {}
        ret["embed_dim"] = cfg.MODEL.OVSEG.EMBED_DIM
        ret["image_size"] = [cfg.INPUT.OVSEG.CROP_SIZE, cfg.INPUT.OVSEG.CROP_SIZE]
        ret["num_masks"] = cfg.MODEL.OVSEG.NUM_MASKS
        return ret

    def _embed_point(self, point: torch.Tensor, feat: torch.Tensor):
        N = len(point)
        if hasattr(point, "tensor"):
            point = point.tensor + 0.5
        point_embed = self.pos_emb.forward_with_coords(
            point.reshape(N, 1, 2), self.image_size
        )
        attr_embed = self.attr_emb(torch.ones_like(point_embed[:, :, 0]).long())
        type_embed = self.point_emb.clone()
        content_embed = feat.unsqueeze(1)
        task_embed = point_embed + type_embed + content_embed + attr_embed

        output_embed = self.mask_emb.repeat(N, 1, 1)[:, -2:]  # 1 x 4 x C
        task_embed = torch.cat([task_embed, output_embed], dim=1)  # N x (1 + 4) x C
        return task_embed, task_embed

    def _embed_multi_point(
        self, points: torch.Tensor, indicator: torch.Tensor, feat: torch.Tensor
    ):
        N, P, _ = points.shape
        points_embed = self.pos_emb.forward_with_coords(points, self.image_size)
        attr_embed = self.attr_emb(indicator.long())
        type_embed = self.point_emb.clone()
        content_embed = feat.unsqueeze(1)
        task_embed = points_embed + type_embed + content_embed + attr_embed

        output_embed = self.mask_emb.repeat(N, 1, 1)[:, 1:2]  # N x 1 x C
        task_embed = torch.cat([task_embed, output_embed], dim=1)  # N x (1 + 1) x C
        return task_embed, task_embed

    def _embed_box(self, box: torch.Tensor, feat: torch.Tensor):
        N = len(box)
        if hasattr(box, "tensor"):
            box = box.tensor + 0.5
        box_embed = self.pos_emb.forward_with_coords(
            box.reshape(N, 2, 2), self.image_size
        )
        corner_embed = self.corner_emb.clone()
        point_embed = self.box_emb.clone()
        content_embed = feat.unsqueeze(1)
        task_embed = box_embed + corner_embed + point_embed + content_embed

        output_embed = self.mask_emb.repeat(N, 1, 1)[:, :1]  # 1 x 1 x C
        task_embed = torch.cat([task_embed, output_embed], dim=1)  # N x (2 + 1) x C
        return task_embed, task_embed

    def forward(
        self,
        points: List[torch.Tensor],
        boxes: List[torch.Tensor],
        points_multi: Tuple[List],
        feats_centers: torch.Tensor,
    ):
        """This is a forward function of embedding multi-type prompts

        Args:
            points (List[torch.Tensor]): A batch of point coordinates. Each one has a shape of [Q, 4].
            feats_centers (torch.Tensor): A batch of center features. It has a shape of [B, Q, C]
        Return:
            List[torch.Tensor]: Prompt embedding has a shape of [B, Q, K, C]
        """
        # embed input prompt into a embedding space
        task_emb, pos_emb = [], []
        if points is not None:
            for pts, feat in zip(points, feats_centers):
                task, pos = self._embed_point(pts, feat)
                task_emb.append(task)
                pos_emb.append(pos)
        if boxes is not None:
            for pts, feat in zip(boxes, feats_centers):
                task, pos = self._embed_box(pts, feat)
                task_emb.append(task)
                pos_emb.append(pos)
        if points_multi is not None:
            for pts, ind, feat in zip(points_multi[0], points_multi[1], feats_centers):
                task, pos = self._embed_point2(pts, ind, feat)
                task_emb.append(task)
                pos_emb.append(pos)

        task_emb = torch.stack(task_emb, dim=0)  # [B, Q, K, C]
        pos_emb = torch.stack(pos_emb, dim=0)  # [B, Q, K, C]

        return task_emb, pos_emb
