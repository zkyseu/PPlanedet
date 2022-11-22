# PPlanedet: A development Toolkit for lane detection based on PaddlePaddle

In this project, we develop a toolkit for lane detection to facilitate research. Especially, PPlanedet is built upon [Paddleseg](https://github.com/PaddlePaddle/PaddleSeg) which is a development Toolkit for segmentation based on PaddlePaddle.

If you do not have enough compute resource, we recommend that you can run our project at [AiStudio](https://aistudio.baidu.com/aistudio/index?ad-from=m-title), which can provide V100 with 32GB memory.

## News 
<ul class="nobull">
  <li>[2022-11-22] :fire: we release the project code. We now only reproduce the <a href="https://arxiv.org/pdf/1712.06080.pdf">SCNN</a> with 93.70% accuracy in Tusimple dataset. Pretrain model will be updated in the following days. We will also release the eval and demo code in the following days.

</ul>

## Introduction
PPlanedet is developed for lane detection based on PaddlPaddle, which is a high performance Deep learning framework. The idea behind the pplanedet is to facilitate researchers who use PaddlePaddle to do research about lane detection. If you have any suggestions about our project, you can contact me.

## Overview

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Models</b>
      </td>
      <td colspan="2">
        <b>Components</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <details><summary><b>Segmentation based</b></summary>
          <ul>
 <li><a href="./configs/pp_liteseg">PP-LiteSeg</a> </li>
            <li><a href="./configs/deeplabv3p">DeepLabV3P</a> </li>
            <li><a href="./configs/ocrnet">OCRNet</a> </li>
            <li><a href="./configs/mobileseg">MobileSeg</a> </li>
            <li><a href="./configs/ann">ANN</a></li>
            <li><a href="./configs/attention_unet">Att U-Net</a></li>
            <li><a href="./configs/bisenetv1">BiSeNetV1</a></li>
            <li><a href="./configs/bisenet">BiSeNetV2</a></li>
            <li><a href="./configs/ccnet">CCNet</a></li>
            <li><a href="./configs/danet">DANet</a></li>
            <li><a href="./configs/ddrnet">DDRNet</a></li>
            <li><a href="./configs/decoupled_segnet">DecoupledSeg</a></li>
            <li><a href="./configs/deeplabv3">DeepLabV3</a></li>
            <li><a href="./configs/dmnet">DMNet</a></li>
            <li><a href="./configs/dnlnet">DNLNet</a></li>
            <li><a href="./configs/emanet">EMANet</a></li>
            <li><a href="./configs/encnet">ENCNet</a></li>
            <li><a href="./configs/enet">ENet</a></li>
            <li><a href="./configs/espnetv1">ESPNetV1</a></li>
            <li><a href="./configs/espnet">ESPNetV2</a></li>
            <li><a href="./configs/fastfcn">FastFCN</a></li>
            <li><a href="./configs/fastscnn">Fast-SCNN</a></li>
            <li><a href="./configs/gcnet">GCNet</a></li>
            <li><a href="./configs/ginet">GINet</a></li>
            <li><a href="./configs/glore">GloRe</a></li>
            <li><a href="./configs/gscnn">GSCNN</a></li>
            <li><a href="./configs/hardnet">HarDNet</a></li>
            <li><a href="./configs/fcn">HRNet-FCN</a></li>
            <li><a href="./configs/hrnet_w48_contrast">HRNet-Contrast</a></li>
            <li><a href="./configs/isanet">ISANet</a></li>
            <li><a href="./configs/pfpn">PFPNNet</a></li>
            <li><a href="./configs/pointrend">PointRend</a></li>
            <li><a href="./configs/portraitnet">PotraitNet</a></li>
            <li><a href="./configs/pp_humanseg_lite">PP-HumanSeg-Lite</a></li>
            <li><a href="./configs/pspnet">PSPNet</a></li>
            <li><a href="./configs/pssl">PSSL</a></li>
            <li><a href="./configs/segformer">SegFormer</a></li>
            <li><a href="./configs/segmenter">SegMenter</a></li>
            <li><a href="./configs/segmne">SegNet</a></li>
            <li><a href="./configs/setr">SETR</a></li>
            <li><a href="./configs/sfnet">SFNet</a></li>
            <li><a href="./configs/stdcseg">STDCSeg</a></li>
            <li><a href="./configs/u2net">U<sup>2</sup>Net</a></li>
            <li><a href="./configs/unet">UNet</a></li>
            <li><a href="./configs/unet_plusplus">UNet++</a></li>
            <li><a href="./configs/unet_3plus">UNet3+</a></li>
            <li><a href="./configs/upernet">UperNet</a></li>
          </ul>
        </details>
        <details><summary><b>Keypoint based</b></summary>
          <ul>
            <li><a href="./EISeg">EISeg</a></li>
            <li>RITM</li>
            <li>EdgeFlow</li>
          </ul>
        </details>
        <details><summary><b>GAN based</b></summary>
          <ul>
            <li><a href="./contrib/PanopticDeepLab/README_CN.md">Panoptic-DeepLab</a></li>
          </ul>
        </details>
      </td>
      <td>
        <details><summary><b>Backbones</b></summary>
          <ul>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/backbones/hrnet.py">HRNet</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/backbones/resnet_cd.py">ResNet</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/backbones/stdcnet.py">STDCNet</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/backbones/mobilenetv2.py">MobileNetV2</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/backbones/mobilenetv3.py">MobileNetV3</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/backbones/shufflenetv2.py">ShuffleNetV2</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/backbones/ghostnet.py">GhostNet</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/backbones/lite_hrnet.py">LiteHRNet</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/backbones/xception_deeplab.py">XCeption</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/backbones/vision_transformer.py">VIT</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/backbones/mix_transformer.py">MixVIT</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/backbones/swin_transformer.py">Swin Transformer</a></li>
          </ul>
        </details>
        <details><summary><b>Losses</b></summary>
          <ul>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/losses/binary_cross_entropy_loss.py">Binary CE Loss</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/losses/bootstrapped_cross_entropy_loss.py">Bootstrapped CE Loss</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/losses/cross_entropy_loss.py">Cross Entropy Loss</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/losses/dice_loss.py">Dice Loss</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/losses/focal_loss.py">Focal Loss</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/losses/binary_cross_entropy_loss.py">MultiClassFocal Loss</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/losses/kl_loss.py">KL Loss</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/losses/l1_loss.py">L1 Loss</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/losses/mean_square_error_loss.py">MSE Loss</a></li>
            <li><a href="https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/paddleseg/models/losses/pixel_contrast_cross_entropy_loss.py">Pixel Contrast CE Loss</a></li>
          </ul>
        </details>
        <details><summary><b>Metrics</b></summary>
          <ul>
            <li>Accuracy</li>
            <li>FP</li>
            <li>FN</li>
          </ul>  
        </details>
      </td>
      <td>
        <details><summary><b>Datasets</b></summary>
          <ul>
            <li><a href="./datasets/tu_simple.py">Tusimple</a></li>  
            <li><a href="./datasets/culane.py">CULane</a></li>
          </ul>
        </details>
        <details><summary><b>Data Augmentation(Paddleseg)</b></summary>
          <ul>
            <li>Flipping</li>  
            <li>Resize</li>  
            <li>ResizeByLong</li>
            <li>ResizeByShort</li>
            <li>LimitLong</li>  
            <li>ResizeRangeScaling</li>  
            <li>ResizeStepScaling</li>
            <li>Normalize</li>
            <li>Padding</li>
            <li>PaddingByAspectRatio</li>
            <li>RandomPaddingCrop</li>  
            <li>RandomCenterCrop</li>
            <li>ScalePadding</li>
            <li>RandomNoise</li>  
            <li>RandomBlur</li>  
            <li>RandomRotation</li>  
            <li>RandomScaleAspect</li>  
            <li>RandomDistort</li>  
            <li>RandomAffine</li>  
          </ul>
        </details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

## License

PPlanedet is released under the [MIT license](LICENSE). We only allow you to use our project for academic uses.

## Citation
If you find our project useful in your research, please consider citing:
    
```latex
@misc{PPlanedet,
    title={PPlanedet, A development Toolkit for lane detection based on PaddlePaddle},
    author={Kunyang Zhou},
    howpublished = {\url{https://github.com/zkyseu/PPlanedet}},
    year={2022}
}
```
    
model reproduced in our project
```latex
@inproceedings{pan2018SCNN,  
  author = {Xingang Pan, Jianping Shi, Ping Luo, Xiaogang Wang, and Xiaoou Tang},  
  title = {Spatial As Deep: Spatial CNN for Traffic Scene Understanding},  
  booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},  
  month = {February},  
  year = {2018}  
}
```
