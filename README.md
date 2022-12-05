# PPlanedet: A Toolkit for lane detection based on PaddlePaddle

In this project, we develop a toolkit for lane detection to facilitate research. Especially, PPlanedet is built upon [Paddleseg](https://github.com/PaddlePaddle/PaddleSeg) which is a development Toolkit for segmentation based on PaddlePaddle.

If you do not have enough compute resource, we recommend that you can run our project at [AiStudio](https://aistudio.baidu.com/aistudio/index?ad-from=m-title), which can provide V100 with 32GB memory for free. We also open source the chinese version at AiStudio. Project link is [here](https://aistudio.baidu.com/aistudio/projectdetail/5099135?contributionType=1)

## News 
<ul class="nobull">
  <li>[2022-12-4] :fire: we release the inference/demo code. You can directly test our model. 
  <li>[2022-11-24] : we release the evaluation code and pretrain weight of the <a href="https://github.com/zkyseu/PPlanedet/tree/main/configs/scnn">SCNN</a> in Tusimple dataset. We also update the Installation and training documentations of our project. In the following days, we will upload Inference/demo code and pretrain weight of SCNN in CULane dataset. Meanwhile, we will reproduce ERFNet.
  <li>[2022-11-22] we release the project code. We now only reproduce the SCNN with 93.70% accuracy in Tusimple dataset. Pretrain model will be updated in the following days. We will also release the eval and demo code in the following days.

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
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/main/configs/scnn">SCNN</a></li>
          </ul>
        </details>
        <details><summary><b>Keypoint based</b></summary>
          <ul>
          </ul>
        </details>
        <details><summary><b>GAN based</b></summary>
          <ul>
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
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/main/dataset/tu_simple.py">Tusimple</a></li>  
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/main/dataset/culane.py">CULane</a></li>
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

## Installation
### step 1 Install PaddlePaddle(you can refer to [official documentation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html))
```Shell
conda create -n pplanedet python=3.8 -y
conda activate pplanedet
conda install paddlepaddle-gpu==2.3.2 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

### step2 Install paddleseg
```Shell
pip install paddleseg
# pip install git+"https://github.com/PaddlePaddle/PaddleSeg"
```

### step3 Install requirements
```Shell
cd PPlanedet
pip install -r requirements.txt
```

## Data preparation
### CULane

Download [CULane](https://xingangpan.github.io/projects/CULane.html). Then extract them to `$CULANEROOT`. Create link to `data` directory.

```Shell
cd $LANEDET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

For CULane, you should have structure like this:
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```

### Tusimple
Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to `$TUSIMPLEROOT`. Create link to `data` directory.

```Shell
cd $LANEDET_ROOT
mkdir -p data
ln -s $TUSIMPLEROOT data/tusimple
```

For Tusimple, you should have structure like this:
```
$TUSIMPLEROOT/clips # data folders
$TUSIMPLEROOT/lable_data_xxxx.json # label json file x4
$TUSIMPLEROOT/test_tasks_0627.json # test tasks json file
$TUSIMPLEROOT/test_label.json # test label json file

```

For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

```Shell
python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT
# this will generate seg_label directory
```

## Getting Started
### Training

For training, run
```Shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python train.py \
        --config configs/scnn/resnet50_tusimple.yaml \
        --do_eval \
        --use_vdl \
        --save_interval 1000 \
        --num_workers 4 \
        --save_dir output
```

```Shell
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch train.py \
        --config configs/scnn/resnet50_tusimple.yaml \
        --do_eval \
        --use_vdl \
        --save_interval 1000 \
        --num_workers 4 \
        --save_dir output
```

### Testing
For testing, run
```Shell
python val.py \
       --config configs/scnn/resnet50_tusimple.yaml \
       --model_path /home/aistudio/181144/pplandet/output/best_model/model.pdparams #please change to your model path
```

### Inference/Demo
```Shell
python predict.py \
       --config configs/scnn/resnet50_tusimple.yaml \
       --model_path your model path
       --image_path your img path
       --custom_color 0 50 100 150 200
```

## License

PPlanedet is released under the [MIT license](LICENSE). We only allow you to use our project for academic uses.

## Acknowledgement
* Thanks [Paddleseg](https://github.com/PaddlePaddle/PaddleSeg) for codebase.
* Some codes are borrowed from [lanedet](https://github.com/Turoad/lanedet).

## Citation
If you find our project useful in your research, please consider citing:
    
```latex
@misc{PPlanedet,
    title={PPlanedet, A Toolkit for lane detection based on PaddlePaddle},
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
