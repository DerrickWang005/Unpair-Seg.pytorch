try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings
    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import argparse
from time import time
from tqdm import tqdm
from glob import glob
import copy
import itertools
import json
import logging
import os
import torch
import torch.nn.functional as F
import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
import lib.models
from lib.utils import add_ovseg_config
from predictor import StageOnePredictor
from utils import calculate_stability_score, remove_small_regions


def get_parser():
    parser = argparse.ArgumentParser(description="Uni-OVSeg inference demo")
    parser.add_argument(
        "-c", "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "uniovseg" module
    logger = setup_logger(output=cfg.OUTPUT_DIR, name="uniovseg")
    return cfg, logger


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg, logger = setup(args)
    logger.info("Arguments: " + str(args))

    predictor = StageOnePredictor(cfg, torch.bfloat16, "cuda")

    if len(args.input) == 1:
        args.input = glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"

    args.input.sort()
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, 'vis'), exist_ok=True)
    for path in args.input:
        start_time = time()
        result = predictor(path)
        image = result.pop("image")
        prediction = result.pop("prediction")
        logger.info(
            "{}: segmented {} instances in {:.2f}s".format(
                path,
                len(prediction),
                time() - start_time,
            )
        )
        # filter - stablility
        stable_score = calculate_stability_score(
            prediction, 0.5, 0.1
        )
        keep = stable_score > 0.92
        prediction = prediction[keep]
        # filter - small disconnected regions and holes
        prediction = prediction.sigmoid().ge(0.5).cpu().numpy().astype(int)
        tmp_masks, scores = [], []
        for pred in prediction:
            mask, changed = remove_small_regions(pred, 15, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, 15, mode="islands")
            unchanged = unchanged and not changed
            tmp_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))
        prediction = torch.cat(tmp_masks, dim=0)

        # visualize
        max_side_len = max(prediction.shape[-2:])
        image = F.interpolate(
            image.unsqueeze(0).float(),
            size=(max_side_len, max_side_len),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        image = (
            image[:, : prediction.shape[1], : prediction.shape[2]]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        vis = Visualizer(image, metadata=None)
        vis.overlay_instances(
            masks=prediction,
            alpha=0.5,
        )
        vis = vis.get_output()
        vis.save(
            os.path.join(
                cfg.OUTPUT_DIR, 'vis', os.path.basename(path)
            )
        )
        del vis
