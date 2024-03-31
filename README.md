# Uni-OVSeg: Open-Vocabulary Segmentation with Unpaired Mask-Text Supervision

This repo contains the code for our paper [Uni-OVSeg](https://derrickwang005.github.io/Uni-OVSeg.pytorch/).
It is a weakly supervised open-vocabulary segmentation framework that leverages unpaired mask-text pairs.

<!-- Our code will be released soon! -->
**Now, we release the inference code and checkpoints for stage one training.**


## Installation
- Linux with Python ≥ 3.10
- PyTorch ≥ 2.1 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- please check `install.sh` for other dependencies


## Inference
The part provides a brief introduction of the usage of Uni-OVSeg.
Please download the [checkpoint](https://drive.google.com/file/d/1LefU25dxFtuPQ5_oA-18_qwKbCQ8wiF9/view?usp=sharing) of stage one training.
We provide `./demo/inference.py` for point-promptable segmentation.
Run it with:

```
cd demo/
python inference.py \
    -c ../configs/S1_point_seg.yaml \
    -i ../images/*.jpg \
    --opt MODEL.WEIGHTS stage1.pth
```

We also provide some test images under `./images/`.

If you use this codebase, or otherwise found our work valuable, please cite:
```
@article{wang2024open,
  title={Open-Vocabulary Segmentation with Unpaired Mask-Text Supervision},
  author={Wang, Zhaoqing and Xia, Xiaobo and Chen, Ziye and He, Xiao and Guo, Yandong and Gong, Mingming and Liu, Tongliang},
  journal={arXiv preprint arXiv:2402.08960},
  year={2024}
}
```
