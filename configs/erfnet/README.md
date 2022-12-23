# ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation

## Abstract
Semantic segmentation is a challenging task that addresses most of the perception needs of Intelligent Vehicles (IV) in an unified way. Deep Neural Networks excel at this task, as they can be trained end-to-end to accurately classify multiple object categories in an image at pixel level. However, a good trade-off between high quality and computational resources is yet not present in state-of-the-art semantic segmentation approaches, limiting their application in real vehicles. In this paper, we propose a deep architecture that is able to run in real-time while providing accurate semantic segmentation. The core of our architecture is a novel layer that uses residual connections and factorized convolutions in order to remain efficient while retaining remarkable accuracy. Our approach is able to run at over 83 FPS in a single Titan X, and 7 FPS in a Jetson TX1 (embedded GPU). A comprehensive set of experiments on the publicly available Cityscapes dataset demonstrates that our system achieves an accuracy that is similar to the state of the art, while being orders of magnitude faster to compute than other architectures that achieve top precision. The resulting
trade-off makes our model an ideal approach for scene understanding in IV applications. The code is publicly available at: https://github.com/Eromera/erfnet

## Model List
| Architecture| Backbone |Dataset | Metric | Config| Checkpoints  |
|-------------|----------|--------|--------|-------|--------------|
| ERFNet      | ERFNet | Tusimple |acc: -| [config](https://github.com/zkyseu/PPlanedet/blob/v3/configs/erfnet/erfnet_tusimple.py)  | [model]|
