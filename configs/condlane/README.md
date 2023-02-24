# CondLaneNet: a Top-to-down Lane Detection Framework Based on Conditional Convolution

## Abstract 
Modern deep-learning-based lane detection methods are successful in most scenarios but struggling for lane lines with complex topologies. In this work, we propose CondLaneNet, a novel top-to-down lane detection framework that detects the lane instances first and then dynamically predicts the line shape for each instance. Aiming to resolve lane instance-level discrimination problem, we introduce a conditional lane detection strategy based on conditional convolution and row-wise formulation. Further, we design the Recurrent Instance Module(RIM) to overcome the problem of detecting lane lines with complex topologies such as dense lines and fork lines. Benefit from the end-to-end pipeline which requires little post-process, our method has real-time efficiency. We extensively evaluate our method on three benchmarks of lane detection. Results show that our method achieves state-of-the-art performance on all three benchmark datasets. Moreover, our method has the coexistence of accuracy and efficiency, e.g. a 78.14 F1 score and 220 FPS on CULane.

## Model List
| Architecture| Backbone |Dataset | Metric | Config| Checkpoints  |
|-------------|----------|--------|--------|-------|--------------|
| CondLaneNet      | ResNet50 | CULane |F1: 79.69| [config](https://github.com/zkyseu/PPlanedet/blob/v4/configs/condlane/resnet50_culane.py)  | [model](https://github.com/zkyseu/PPlanedet/releases/download/CondLaneNewt/model.pd)|
| CondLaneNet|ConvNext|CULane|F1: 75.20| [config](https://github.com/zkyseu/PPlanedet/blob/v4/configs/condlane/convnext_culane.py) |[model](https://github.com/zkyseu/PPlanedet/releases/download/convnext/model.pd)
|CondaLaneNet|CSPResNet50|CULane|F1: 79.92|[config](https://github.com/zkyseu/PPlanedet/blob/v5/configs/condlane/cspresnet_50_culane.py)|[model]

Note: We train CondLaneNet with ConvNext from scratch. If you want to obtain higher performance with convnext, we suggest that you can load imagenet pretrain weight for convnext. The pretrain weight of convnext can be found in Paddleclas
