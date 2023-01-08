# RTFormer: Efficient Design for Real-Time Semantic Segmentation with Transformer

## Abstract
Recently, transformer-based networks have shown impressive results in semantic segmentation. Yet for real-time semantic segmentation, pure CNN-based approaches still dominate in this field, due to the time-consuming computation
mechanism of transformer. We propose RTFormer, an efficient dual-resolution transformer for real-time semantic segmenation, which achieves better trade-off between performance and efficiency than CNN-based models. To achieve high
inference efficiency on GPU-like devices, our RTFormer leverages GPU-Friendly Attention with linear complexity and discards the multi-head mechanism. Besides, we find that cross-resolution attention is more efficient to gather global context information for high-resolution branch by spreading the high level knowledge learned
from low-resolution branch. Extensive experiments on mainstream benchmarks demonstrate the effectiveness of our proposed RTFormer, it achieves state-of-the-art on Cityscapes, CamVid and COCOStuff, and shows promising results on ADE20K.

## Model List
| Architecture| Backbone |Dataset | Metric | Config| Checkpoints  |
|-------------|----------|--------|--------|-------|--------------|
|RTformer      | RTformer | Tusimple |acc: 96.55| [config](https://github.com/zkyseu/PPlanedet/blob/v3/configs/rtformer/rtformer_tusimple.py)  | [model](https://github.com/zkyseu/PPlanedet/releases/download/rtformer/epoch_46.pd)|
