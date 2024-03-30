import os.path as osp
import logging
from functools import partial
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.modeling import build_model


def inference_transform(img_path, transform):
    output = dict()

    # load image
    image = utils.read_image(img_path, format="RGB")
    ori_shape = image.shape[:2]
    output["height"] = ori_shape[0]
    output["width"] = ori_shape[1]

    # preprocess
    image, _ = T.apply_transform_gens(transform, image)
    output["image"] = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))

    return output


class StageOnePredictor:
    def __init__(self, cfg, dtype=torch.bfloat16, device="cuda"):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        print("Model:\n{}".format(self.model))

        checkpointer = DetectionCheckpointer(self.model)
        msg = checkpointer.load(cfg.MODEL.WEIGHTS)
        print(msg)

        augments = [
            T.ResizeShortestEdge(cfg.INPUT.CROP_SIZE, cfg.INPUT.CROP_SIZE),
            T.FixedSizeCrop(
                crop_size=(cfg.INPUT.CROP_SIZE, cfg.INPUT.CROP_SIZE), seg_pad_value=0
            ),
        ]
        self.transform = partial(inference_transform, transform=augments)

        self.dtype = dtype
        self.device = device

    @torch.inference_mode()
    def __call__(self, image_path):
        image = self.transform(image_path)
        image["image"] = image["image"].to(self.device)
        with torch.cuda.amp.autocast(
            enabled=self.dtype == torch.bfloat16, dtype=self.dtype
        ):
            prediction = self.model([image])[0]["proposal"]
        image["prediction"] = prediction
        return image
