# PPlanedet: A Toolkit for lane detection based on PaddlePaddle

In this project, we develop a toolkit for lane detection to facilitate research. Welcome to join us to make this project more perfect and practical.

If you do not have enough compute resource, we recommend that you can run our project at [AiStudio](https://aistudio.baidu.com/aistudio/index?ad-from=m-title), which can provide you with V100(32GB memory) for free. We also opened source the chinese version at AiStudio. Project link is [here](https://aistudio.baidu.com/aistudio/projectdetail/5316470?contributionType=1)

## News 
If you want to learn more changes, you can refer to [History](https://github.com/zkyseu/PPlanedet/blob/v4/file/change_log.md).
<ul class="nobull">
  <li>[2023-02-09] : We updated some common models in pplanedet and released the convnext backbone. In the following days, we will focus on solving bugs in CLRNet.
  <li>[2023-02-06] : We fixed some bugs in detect.py and added some data augmentation methods（e.g.GuassianBulr).
  <li>[2023-01-20] : We released the code of CLRNet. However, there still exists some bugs(NMS cuda code meets error) in CLRNet during inference. We will fix this bug soon.
  <li>[2023-01-16] :fire: We released the version4. In v4, we reproduced the <a href="https://github.com/zkyseu/PPlanedet/tree/v4/configs/condlane">CondLane</a>, a state-of-the-art lane detection method based on Keypoint. Meanwhile, we fixed some bugs in albumentations and made the albumentations be able to used in Keypoint based method. In order to facilitate the scientific research, we reproduced the <a href="https://github.com/zkyseu/PPlanedet/tree/v4/configs/rtformer">RTFormer</a>, a SOTA real-time semantic segmentation with Transformer.


</ul>

## Introduction
PPlanedet is developed for lane detection based on PaddlPaddle. PaddlePaddle is a high performance Deep learning framework. The idea behind the pplanedet is to facilitate researchers who use PaddlePaddle to do research about lane detection. If you have any suggestions about our project, you can contact me.

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
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v4/configs/rtformer">RTFormer</a></li>
          </ul>
        </details>
        <details><summary><b>Keypoint(anchor) based</b></summary>
          <ul>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v3/configs/ufld">UFLD</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v4/configs/condlane">CondLane</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v4/configs/clrnet">CLRNet</a></li>
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
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v4/pplanedet/model/backbones/convnext.py">ConvNext</a></li>
          </ul>
        </details>
        <details><summary><b>Losses</b></summary>
          <ul>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v2/pplanedet/model/losses/binary_cross_entropy_loss.py">Binary CE Loss</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v2/pplanedet/model/losses/cross_entropy_loss.py">Cross Entropy Loss</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v2/pplanedet/model/losses/focal_loss.py">Focal Loss</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v2/pplanedet/model/losses/focal_loss.py">MultiClassFocal Loss</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v4/pplanedet/model/losses/regl1_loss.py">RegL1KpLoss</a></li>
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
            <li>Colorjitters</li>
            <li>RandomErasings</li>
            <li>GaussianBlur</li>
            <li>RandomGrayScale</li>
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
# first you should add 'seg = False' in your config 
python tools/detect.py configs/scnn/resnet50_tusimple.py --img images\
          --load_from model.pd --savedir ./vis
```

If you want to save the visualization of the segmentation results, you can run the following code
```
# first you should add 'seg = True' in your config 
python tools/detect.py configs/scnn/resnet50_tusimple.py --img images\
          --load_from model.pd --savedir ./vis
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
@Inproceedings{pan2018SCNN,  
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

@article{2017ERFNet,
author = {E.Romera, J.M.Alvarez, L.M.Bergasa and R.Arroyo},
title = {ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation},
journal = {IEEE Transactions on Intelligent Transportation Systems(T-ITS)},
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

@InProceedings{CondLaneNet,
author = {Liu, Lizhe and Chen, Xiaohao and Zhu, Siyu and Tan, Ping},
title = {CondLaneNet: a Top-to-down Lane Detection Framework Based on Conditional Convolution},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
year = {2021}
}

@article{RTFormer,
author = {Wang, Jian, Chenhui Gou, Qiman Wu, Haocheng Feng, Junyu Han, Errui Ding, and Jingdong Wang},
title = {RTFormer: Efficient Design for Real-Time Semantic Segmentation with Transformer},
journal = {arXiv preprint arXiv:2210.07124 (2022)},
year = {2022}
}
```
