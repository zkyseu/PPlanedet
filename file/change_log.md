## History of PPLandet

In this file, we show the development of the PPLanedet.

## History
<ul class="nobull">
  <li>[2023-01-20] : We released the code of CLRNet.
  <li>[2023-01-16] :fire: We released the version4. In v4, we reproduced the <a href="https://github.com/zkyseu/PPlanedet/tree/v4/configs/condlane">CondLane</a>, a state-of-the-art lane detection method based on Keypoint. Meanwhile, we fixed some bugs in albumentations and made the albumentations be able to used in Keypoint based method. In order to facilitate the scientific research, we reproduced the <a href="https://github.com/zkyseu/PPlanedet/tree/v4/configs/rtformer">RTFormer</a>, a SOTA real-time semantic segmentation with Transformer.
  <li>[2023-01-01] : We released the <a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/deeplabv3p">DeepLabV3+</a>. Version 4 is coming soon. We will also open source the Colab demo of the PPLanedet.
  <li>[2022-12-29] : We reproduced <a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/resa">RESA</a> and pretrain weight is available. We also fixed some bugs in detect.py
  <li>[2022-12-23] : We reproduced the <a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/erfnet">ERFNet</a> and the weight is available. We also support the visualization of the segmentation results and code is shown in <a href="https://github.com/zkyseu/PPlanedet/blob/v3/tools/detect.py">detect.py</a>
  <li>[2022-12-19] : Versionv3 has been released. In this version, we reproduce the <a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/ufld">UFLD</a> and the pretrain weight is available.
  <li>[2022-12-14] : We released versionv2. Compared with v1, v2 is achieved by Hook instead of being built upon Paddleseg. With v2, we can obtain a better SCNN with 95% accuracy on Tusimple dataset. It should be noticed that we only spent 30 epochs to achieve this result. Pretrain weight is available.
  <li>[2022-12-4] : We released the inference/demo code. You can directly test our model. 
  <li>[2022-11-24] : We released the evaluation code and pretrain weight of the <a href="https://github.com/zkyseu/PPlanedet/tree/main/configs/scnn">SCNN</a> in Tusimple dataset. We also updated the Installation and training documentations of our project. In the following days, we will upload Inference/demo code and pretrain weight of SCNN in CULane dataset. Meanwhile, we will reproduce ERFNet.
  <li>[2022-11-22] We released the project code. We now only reproduce the SCNN with 93.70% accuracy in Tusimple dataset. Pretrain model will be updated in the following days. We will also release the eval and demo code in the following days.
