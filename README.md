# PPlanedet: A Toolkit for lane detection based on PaddlePaddle

In this project, we develop a toolkit for lane detection to facilitate research. Welcome to join us to make this project more perfect and practical.

If you do not have enough compute resource, we recommend that you can run our project at [AiStudio](https://aistudio.baidu.com/aistudio/index?ad-from=m-title), which can provide you with V100(32GB memory) for free. We also open source the chinese version at AiStudio. Project link is [here](https://aistudio.baidu.com/aistudio/projectdetail/5316470?contributionType=1)

## News 
<ul class="nobull">
  <li>[2023-01-16] :fire: We released the version4. In v4, we reproduced the CondLane, a state-of-the-art lane detection method based on Keypoint and weight is coming soon. Meanwhile, we fixed some bugs in "Alug" augmentation and made the data preprocessing be able to used in Keypoint based method. In order to achieve high performance, we provided RTFormer, a SOTA real-time segmentation Transformer, to facilitate developers to choose the model.
  <li>[2023-01-01] : We released the <a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/deeplabv3p">DeepLabV3+</a>. Version 4 is coming soon. We will also open source the Colab demo of the PPLanedet.
  <li>[2022-12-29] : We reproduced <a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/resa">RESA</a> and pretrain weight is available. We also fixed some bugs in detect.py
  <li>[2022-12-23] : We reproduced the <a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/erfnet">ERFNet</a> and the weight is available. We also support the visualization of the segmentation results and code is shown in <a href="https://github.com/zkyseu/PPlanedet/blob/v3/tools/detect.py">detect.py</a>
  <li>[2022-12-19] : Versionv3 has been released. In this version, we reproduce the <a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/ufld">UFLD</a> and the pretrain weight is available.
  <li>[2022-12-14] : We released versionv2. Compared with v1, v2 is achieved by Hook instead of being built upon Paddleseg. With v2, we can obtain a better SCNN with 95% accuracy on Tusimple dataset. It should be noticed that we only spent 30 epochs to achieve this result. Pretrain weight is available.
  <li>[2022-12-4] : We released the inference/demo code. You can directly test our model. 
  <li>[2022-11-24] : We released the evaluation code and pretrain weight of the <a href="https://github.com/zkyseu/PPlanedet/tree/main/configs/scnn">SCNN</a> in Tusimple dataset. We also updated the Installation and training documentations of our project. In the following days, we will upload Inference/demo code and pretrain weight of SCNN in CULane dataset. Meanwhile, we will reproduce ERFNet.
  <li>[2022-11-22] We released the project code. We now only reproduce the SCNN with 93.70% accuracy in Tusimple dataset. Pretrain model will be updated in the following days. We will also release the eval and demo code in the following days.

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
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/resa">RESA</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/main/configs/scnn">SCNN</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/erfnet">ERFNet</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/deeplabv3p">DeepLabV3+</a></li>
          </ul>
        </details>
        <details><summary><b>Keypoint based</b></summary>
          <ul>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/ufld">UFLD</a></li>
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
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v3/pplanedet/model/backbones/resnet.py">ResNet</a></li>
          </ul>
        </details>
        <details><summary><b>Losses</b></summary>
          <ul>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v2/pplanedet/model/losses/binary_cross_entropy_loss.py">Binary CE Loss</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v2/pplanedet/model/losses/cross_entropy_loss.py">Cross Entropy Loss</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v2/pplanedet/model/losses/focal_loss.py">Focal Loss</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v2/pplanedet/model/losses/focal_loss.py">MultiClassFocal Loss</a></li>
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
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v2/pplanedet/datasets/tusimple.py">Tusimple</a></li>  
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v2/pplanedet/datasets/culane.py">CULane</a></li>
          </ul>
        </details>
        <details><summary><b>Data Augmentation</b></summary>
          <ul>
            <li>RandomLROffsetLABEL</li>  
            <li>Resize</li>  
            <li>RandomUDoffsetLABEL</li>
            <li>RandomCrop</li>
            <li>CenterCrop</li>  
            <li>RandomRotation</li>  
            <li>RandomBlur</li>
            <li>Normalize</li>
            <li>RandomHorizontalFlip</li>
            <li>Alaug</li> 
          </ul>
        </details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

## Installation
### step 1 Install PaddlePaddle>=2.4.0(you can refer to [official documentation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html))
```Shell
conda create -n pplanedet python=3.8 -y
conda activate pplanedet
conda install paddlepaddle-gpu==2.4.1 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

### step2 Git clone PPlanedet
```Shell
git clone https://github.com/zkyseu/PPlanedet
```

### step3 Install PPlanedet
```Shell
cd PPlanedet
pip install -r requirements.txt
python setup.py build develop
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
# python tools/generate_seg_tusimple.py --root /root/paddlejob/workspace/train_data/datasets --savedir /root/paddlejob/workspace/train_data/datasets/seg_label
```

## Getting Started
### Training

For training, run(shell scripts are under folder script). More training details are in [documentation](https://github.com/zkyseu/PPlanedet/blob/v3/DOC.md)
```Shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/scnn/resnet50_tusimple.py
```

```Shell
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch tools/train.py -c configs/scnn/resnet50_tusimple.py
```

### Testing
For testing, run
```Shell
python tools/train.py -c configs/scnn/resnet50_tusimple.py \
                      --load /home/fyj/zky/tusimple/new/pplanedet/output_dir/resnet50_tusimple/latest.pd \
                      --evaluate-only 
```

### Inference/Demo
See `tools/detect.py` for detailed information.
```
python tools/detect.py --help

usage: detect.py [-h] [--img IMG] [--show] [--savedir SAVEDIR]
                 [--load_from LOAD_FROM]
                 config

positional arguments:
  config                The path of config file

optional arguments:
  -h, --help            show this help message and exit
  --img IMG             The path of the img (img file or img_folder), for
                        example: data/*.png
  --show                Whether to show the image
  --savedir SAVEDIR     The root of save directory
  --load_from LOAD_FROM
                        The path of model
```
To run inference on example images in `./images` and save the visualization images in `vis` folder:
```
python tools/detect.py configs/scnn/resnet50_tusimple.py --img images\
          --load_from model.pth --savedir ./vis
```

If you want to save the visualization of the segmentation results, you can run the following code
```
# first you should add 'seg = True' in your config 
python tools/detect.py configs/scnn/resnet50_tusimple.py --img images\
          --load_from model.pth --savedir ./vis
```

## License

PPlanedet is released under the [MIT license](LICENSE). We only allow you to use our project for academic uses.

## Acknowledgement
* Thanks [PASSL](https://github.com/PaddlePaddle/PASSL) for providing Hook codes
* Thanks [lanedet](https://github.com/Turoad/lanedet) for providing model codes.

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

@InProceedings{qin2020ultra,
author = {Qin, Zequn and Wang, Huanyu and Li, Xi},
title = {Ultra Fast Structure-aware Deep Lane Detection},
booktitle = {The European Conference on Computer Vision (ECCV)},
year = {2020}
}

@InProceedings{2017ERFNet,
author = {E.Romera, J.M.Alvarez, L.M.Bergasa and R.Arroyo},
title = {ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation},
booktitle = {IEEE Transactions on Intelligent Transportation Systems(T-ITS)},
year = {2017}
}

@InProceedings{2021RESA,
author = {Zheng, Tu and Fang, Hao and Zhang, Yi and Tang, Wenjian and Yang, Zheng and Liu, Haifeng and Cai, Deng},
title = {RESA: Recurrent Feature-Shift Aggregator for Lane Detection},
booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
year = {2021}
}

@InProceedings{DeepLabV3+,
author = {Chen, Liang-Chieh, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam},
title = {Encoder-decoder with atrous separable convolution for semantic image segmentation},
booktitle = {In Proceedings of the European conference on computer vision(ECCV)},
year = {2018}
}
```
