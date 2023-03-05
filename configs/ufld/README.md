# Ultra Fast Structure-aware Deep Lane Detection

## Abstract
Modern methods mainly regard lane detection as a problem of pixel-wise segmentation, which is struggling to address the problem of challenging scenarios and speed. Inspired by human perception, the recognition of lanes under severe occlusion and extreme lighting conditions is mainly based on contextual and global information. Motivated by this observation, we propose a novel, simple, yet effective formulation aiming at extremely fast speed and challenging scenarios. Specifically, we treat the process of lane detection as a row-based selecting problem using global features. With the help of row-based selecting, our formulation could significantly reduce the computational cost. Using a large receptive field on global features, we could also handle the challenging scenarios. Moreover, based on the formulation, we also propose a structural loss to explicitly model the structure of lanes. Extensive experiments on two lane detection benchmark datasets show that our method could achieve the state-of-the-art performance in terms of both speed and accuracy. A light-weight version could even achieve 300+ frames per second with the same resolution, which is at least 4x faster than previous state-of-the-art methods. Our code will be made publicly available.

## Models
| Architecture| Backbone |Dataset | Metric | Config| Checkpoints  |
|-------------|----------|--------|--------|-------|--------------|
| UFLD | ResNet18 | Tusimple |acc: 94.78| [config](resnet18_tusimple.py)  | [model](https://github.com/zkyseu/PPlanedet/releases/download/UFLD/epoch_96.pd)|
|UFLD|MobileNetV3|Tusimple| acc: 95.71|[config](mobilenetv3_tusimple.py)|[model](https://github.com/zkyseu/PPlanedet/releases/download/UFLD/model.pd)|
|UFLD|CSPResNet-m|Tusimple|acc: 96.05|[config](cspresnet_tusimple.py)|[model](https://github.com/zkyseu/PPlanedet/releases/download/UFLD/cspresnet.pd)
|UFLD|MobileNetV3|CULane|F1: 66.25|[config](mobilenetv3_culane.py)|[model](https://github.com/zkyseu/PPlanedet/releases/download/UFLD/model_culane.pd)
