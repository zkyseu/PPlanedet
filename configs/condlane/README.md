# CondLaneNet: a Top-to-down Lane Detection Framework Based on Conditional Convolution

## Abstract 
Modern deep-learning-based lane detection methods are successful in most scenarios but struggling for lane lines with complex topologies. In this work, we propose CondLaneNet, a novel top-to-down lane detection framework that detects the lane instances first and then dynamically predicts the line shape for each instance. Aiming to resolve lane instance-level discrimination problem, we introduce a conditional lane detection strategy based on conditional convolution and row-wise formulation. Further, we design the Recurrent Instance Module(RIM) to overcome the problem of detecting lane lines with complex topologies such as dense lines and fork lines. Benefit from the end-to-end pipeline which requires little post-process, our method has real-time efficiency. We extensively evaluate our method on three benchmarks of lane detection. Results show that our method achieves state-of-the-art performance on all three benchmark datasets. Moreover, our method has the coexistence of accuracy and efficiency, e.g. a 78.14 F1 score and 220 FPS on CULane.

## Model List
| Architecture| Backbone |Dataset | Metric | Config| Checkpoints  |
|-------------|----------|--------|--------|-------|--------------|
| CondLaneNet      | ResNet50 | CULane |F1: 79.69| [config](resnet50_culane.py)  | [model](https://github.com/zkyseu/PPlanedet/releases/download/CondLaneNewt/model.pd)|
| CondLaneNet|ConvNext|CULane|F1: 75.20| [config](convnext_culane.py) |[model](https://github.com/zkyseu/PPlanedet/releases/download/convnext/model.pd)
|CondLaneNet|CSPResNetm|CULane|F1: 79.92|[config](cspresnet_50_culane.py)|[model](https://github.com/zkyseu/PPlanedet/releases/download/CondLaneNewt/model_csp.pd)

### Note
1. We train CondLaneNet with ConvNext from scratch. If you want to obtain higher performance with convnext, we suggest that you can load imagenet pretrain weight for convnext. The pretrain weight of convnext can be found in Paddleclas.


2. Our CondLaneNet achieves 79.92 F1 score on CULane dataset. In our CondLaneNet, we ablate some components in CondLaneNet like backbone and neck. We replace the backbone with CSPResNet which has less parameters and higher performance than ResNet family. In addition, we introduce CSPSimSPPF in YOLOv6 to replace Self attention in CondLaneNet. CSPSimSPPF needs less compute resource, less parameters and higher performance than self attention. Finally, we replace FPN in CondLaneNet with CSPRepBiFPN to further improve model performance. We also adopt EMA to train CondLaneNet. Compared with origin CondLaneNet, our CondLaneNet has less parameters(11M vs 23M) and achieved higher performance(79.92 vs 79.69). Compared with SOTA method CLRNet(ResNet34), our CondLaneNet also realizes better performance(79.92 vs 79.72) and less parameters(11M vs 21M).
