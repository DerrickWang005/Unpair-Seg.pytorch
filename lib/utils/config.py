from detectron2.config import CfgNode as CN


def add_ovseg_config(cfg):
    """
    Add config for uniovseg.
    """
    cfg.INPUT.DATASET_ROOT = "/datasets/sharegpt4v"
    cfg.INPUT.DATASET_URL = [
        ["/datasets/SA-1B/split1-2m", "datadict_0p5.parquet"],
    ]
    cfg.INPUT.DATASET_JSON = '/vepfs/home/wangzhaoqing/uni-ovseg/sa1b.json'
    cfg.INPUT.FEW_SHOT_JSON = [
        "/workspace/pretrains/coco_fewshot/openvocab_coco_2017_train_panoptic_with_sem_seg_0.1.json",
    ]
    cfg.INPUT.DATASET_MAPPER_NAME = "sa1b"
    cfg.INPUT.COLOR_AUG_SSD = True
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.INPUT.SIZE_DIVISIBILITY = 32
    cfg.INPUT.IMG_SIZE = 1024
    cfg.INPUT.CROP_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.8
    cfg.INPUT.MAX_SCALE = 1.2
    cfg.INPUT.MIN_AREA_RATIO = 0.001
    cfg.INPUT.MAX_AREA_RATIO = 0.8
    cfg.INPUT.MAX_INSTANCE = 40

    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0
    cfg.SOLVER.POLY_LR_POWER = 0.9

    cfg.MODEL.META_ARCHITECTURE = "UniOVSeg_S1"
    cfg.MODEL.OVSEG = CN()
    cfg.MODEL.OVSEG.CLIP_MODEL_NAME = "convnext_large_d_320"
    cfg.MODEL.OVSEG.CLIP_PRETRAINED_WEIGHTS = (
        "/workspace/pretrains/convnext_large_d_320.laion2B-s29B-b131K-ft-soup.pth"
    )
    cfg.MODEL.OVSEG.AUX_MODEL_NAME = "convnext_xxlarge"
    cfg.MODEL.OVSEG.AUX_PRETRAINED_WEIGHTS = (
        "/workspace/pretrains/convnext_xxlarge.laion2B-s34B-b82K-augreg-soup.pth"
    )
    cfg.MODEL.OVSEG.PROMPT_ENCODER_NAME = "PromptEncoder"
    cfg.MODEL.OVSEG.PIXEL_DECODER_NAME = "BasePixelDecoder"
    cfg.MODEL.OVSEG.TRANSFORMER_ENC_LAYERS = 6
    cfg.MODEL.OVSEG.IN_FEATURES = [
        "res2",
        "res3",
        "res4",
        "res5",
    ]
    cfg.MODEL.OVSEG.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = [
        "res3",
        "res4",
        "res5",
    ]
    cfg.MODEL.OVSEG.COMMON_STRIDE = 4
    cfg.MODEL.OVSEG.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    cfg.MODEL.OVSEG.TRANSFORMER_DECODER_NAME = "MultiScaleMaskDecoder"
    cfg.MODEL.OVSEG.MASK_DIM = 256
    cfg.MODEL.OVSEG.CONVS_DIM = 256
    cfg.MODEL.OVSEG.NORM = "GN"
    cfg.MODEL.OVSEG.EMBED_DIM = 256
    cfg.MODEL.OVSEG.NHEADS = 8
    cfg.MODEL.OVSEG.DIM_FEEDFORWARD = 2048
    cfg.MODEL.OVSEG.PRE_NORM = False
    cfg.MODEL.OVSEG.DROPOUT = 0.0
    cfg.MODEL.OVSEG.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.OVSEG.DEEP_SUPERVISION = True
    cfg.MODEL.OVSEG.NUM_MASKS = 4
    cfg.MODEL.OVSEG.RANK = 8
    cfg.MODEL.OVSEG.LORA_INIT = False
    cfg.MODEL.OVSEG.CRITERION_SEG = "Many2ManySetCriterion"
    cfg.MODEL.OVSEG.CRITERION_ALIGN = "MaskTextAlignCriterion"
    cfg.MODEL.OVSEG.DICE_WEIGHT = 1.0
    cfg.MODEL.OVSEG.MASK_WEIGHT = 1.0
    cfg.MODEL.OVSEG.IOU_WEIGHT = 1.0
    cfg.MODEL.OVSEG.ALIGN_WEIGHT = 1.0
    cfg.MODEL.OVSEG.MATCHER_NUM_POINTS = 5000
    cfg.MODEL.OVSEG.MATCHER_THRES_POS = 0.7
    cfg.MODEL.OVSEG.TRAIN_NUM_POINTS = 12544  # 800 * 800 // (8 * 8)
    cfg.MODEL.OVSEG.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.OVSEG.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.OVSEG.DEC_LAYERS = 7
    cfg.MODEL.OVSEG.LOSS_TOPK = 1.0
    cfg.MODEL.OVSEG.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = True
    cfg.MODEL.OVSEG.PTS_PER_SIDE = (10, 10)
    cfg.MODEL.OVSEG.CLIP_DIM = 1024
    cfg.MODEL.OVSEG.SIZE_DIVISIBILITY = 32
    cfg.MODEL.OVSEG.INPUT_SIZES = [896, 1024]

    cfg.MODEL.OVSEG.TEST = CN()
    cfg.MODEL.OVSEG.TEST.PTS_PER_SIDE = (20, 20)
    cfg.MODEL.OVSEG.TEST.OBJECT_MASK_THRESHOLD = 0.5
    cfg.MODEL.OVSEG.TEST.OVERLAP_THRESHOLD = 0.5
    cfg.MODEL.OVSEG.TEST.SEMANTIC_ON = False
    cfg.MODEL.OVSEG.TEST.INSTANCE_ON = False
    cfg.MODEL.OVSEG.TEST.PANOPTIC_ON = False
    cfg.MODEL.OVSEG.TEST.MASKCLS_ON = False
    cfg.MODEL.OVSEG.TEST.AUTOLABEL_ON = True
    cfg.MODEL.OVSEG.TEST.AUTOLABEL_TYPE = "panoptic-point"
    cfg.MODEL.OVSEG.TEST.AUTOLABEL_SAVE = False
