<font size=4> 简体中文 | [English](ENGLISH.md)
## 🚀PPLanedet: 基于PaddlePaddle的车道线检测的工具包

<font size=3> 在这个项目中，我们开发了PPLanedet用于车道线检测。PPLanedet中包含了很多先进的车道线检测算法以方便车道线检测的科学研究和工程应用。欢迎加入我们来完善PPLanedet。如果您觉得PPLanedet不错，可以给我们项目一个star。

<font size=3>如果您没有充足的计算资源，我们建议您可以在百度[AiStudio](https://aistudio.baidu.com/aistudio/index?ad-from=m-title)上运行PPLanedet。在Aistudio上您可以免费获得V100、A100两种高性能GPU计算资源，我们也在AIstudio上公开了PPLanedet的运行demo，项目链接在[这里](https://aistudio.baidu.com/aistudio/projectdetail/5316470?contributionType=1)。由于在Windows系统上可能存在权限问题，我们建议您在linux系统上运行PPLanedet。

## 🆕新闻
在这个部分中，我们展示PPLanedet中最新的改进，如果您想要了解更多关于PPLanedet的改动，您可以浏览[改动历史](https://github.com/zkyseu/PPlanedet/blob/v5/file/change_log.md)。
<ul class="nobull">
  <li>[2023-05-01] : 我们基于DETR提出一个端到端的车道线检测模型<a href="https://github.com/zkyseu/O2SFormer">O2SFormer</a>, 欢迎大家使用！
  <li>[2023-03-05] : 我们公开了UFLD模型在CULane上的预训练权重，并且增加了模型导出为预训练格式的功能。
  <li>[2023-03-04] : 我们在V5中增加了Visualdl可视化功能，VisualDL功能类似tensorboard。后续我们会完善PPLanedet文档关于如何在PPLanedet中增加组件，如果想尝试可以参考<a href="https://github.com/open-mmlab/mmdetection/blob/master/docs/en/tutorials/customize_models.md">mmdetection</a>。
  <li>[2023-03-01] : 我们修改了PPLanedet中的一些bug，目前CLRNet还在调试中，如果您想获得高性能的车道线检测模型，我们建议您可以使用我们改进的CondLaneNet。
  <li>[2023-02-24] :fire: 我们发布了PPLanedet的第五个版本(version5)。在V5中，我们复现了更多实用的backbone和Neck等模块(例如YOLOv6中的CSPRepBiFPN、CSPSimSPPF)。依靠这些更加先进的模块，我们得到了一个性能更佳的CondLaneNet。改进的CondLaneNet在CULane数据集上达到79.92的F1 score并且参数量只有11M，更多的细节可以参考CondLaneNet的<a href="https://github.com/zkyseu/PPlanedet/tree/v5/configs/condlane">配置文件</a>。 

</ul>

## 👀介绍
PPLanedet是一个基于PaddlePaddle的车道线检测工具包。PaddlePaddle是一种高性能的深度学习框架。PPLanedet开发的初衷是希望科研人员或者工程师能够通过一个框架方便地开发各类车道线检测算法。如果您对PPLanedet有任何疑问或者建议，欢迎和我联系。

## 🌟框架总览

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>模型</b>
      </td>
      <td colspan="2">
        <b>框架组件</b>
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
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v5/configs/ufld">UFLD</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v5/configs/condlane">CondLane</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/tree/v5/configs/clrnet">CLRNet</a></li>
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
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v4/pplanedet/model/backbones/mobilenet.py">MobileNetV3</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v4/pplanedet/model/backbones/cspresnet.py">CSPResNet</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v5/pplanedet/model/backbones/shufflenet.py">ShuffleNet</a></li>
          </ul>
        </details>
        <details><summary><b>Necks</b></summary>
          <ul>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v5/pplanedet/model/necks/fpn.py">FPN</a></li>
            <li><a href="https://github.com/zkyseu/PPlanedet/blob/v5/pplanedet/model/necks/csprepbifpn.py">CSPRepbifpn</a></li>
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

## 🛠️安装
### 步骤1 安装 PaddlePaddle>=2.4.0(如果有疑问可以参考[官方文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html))
```Shell
conda create -n pplanedet python=3.8 -y
conda activate pplanedet
conda install paddlepaddle-gpu==2.4.1 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

### 步骤2 Git clone PPlanedet
```Shell
git clone https://github.com/zkyseu/PPlanedet
```

### 步骤3 安装 PPlanedet
```Shell
cd PPlanedet
pip install -r requirements.txt
python setup.py build develop
```

## 📘数据集准备(CULane和Tusimple为例)
### CULane

下载 [CULane](https://xingangpan.github.io/projects/CULane.html). 接着解压到 `$CULANEROOT`. 创建 `data` 目录.

```Shell
cd $LANEDET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

对于CULane数据集, 完成以上步骤你应该有下列数据集结构:
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```

### Tusimple
下载 [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). 然后解压到 `$TUSIMPLEROOT`. 创建 `data` 文件夹.

```Shell
cd $LANEDET_ROOT
mkdir -p data
ln -s $TUSIMPLEROOT data/tusimple
```

对于Tusimple数据集, 完成以上步骤你应该有下列数据集结构:
```
$TUSIMPLEROOT/clips # data folders
$TUSIMPLEROOT/lable_data_xxxx.json # label json file x4
$TUSIMPLEROOT/test_tasks_0627.json # test tasks json file
$TUSIMPLEROOT/test_label.json # test label json file

```

对于Tusimple数据集，分割地标签并没有提供，因此为了方便分割模型的训练，我们运行下列命令从json文件中生成分割的mask。 

```Shell
python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT
# python tools/generate_seg_tusimple.py --root /root/paddlejob/workspace/train_data/datasets --savedir /root/paddlejob/workspace/train_data/datasets/seg_label
```
### 自制数据集
如果你想在自己数据集上进行训练，我们在[issue #1](https://github.com/zkyseu/PPlanedet/issues/1)中对该问题进行了讨论，大家可以进行参考

## 💎开始快乐炼丹
### 1、训练的命令
对于训练, 运行以下命令(shell脚本在script文件夹下)。更多的训练命令可以参考[documentation](https://github.com/zkyseu/PPlanedet/blob/v3/DOC.md)
```Shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/scnn/resnet50_tusimple.py
```

多卡训练(基于分割的模型可以稳定运行，其他模型训练还不太稳定)
```Shell
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch tools/train.py -c configs/scnn/resnet50_tusimple.py
```

### 2、测试
运行以下命令开启模型的测试
```Shell
python tools/train.py -c configs/scnn/resnet50_tusimple.py \
                      --load /home/fyj/zky/tusimple/new/pplanedet/output_dir/resnet50_tusimple/latest.pd \
                      --evaluate-only 
```

### 3、推理/Demo
想了解更多细节，请参考 `tools/detect.py`.
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
运行以下命令对一个文件夹下的图片进行预测，可视化结果保存在文件夹 `vis`下，如果您的模型不是分割模型，需要在配置文件中加上 seg=False，具体可见[issue3](https://github.com/zkyseu/PPlanedet/issues/3)
```
# first you should add 'seg = False' in your config 
python tools/detect.py configs/scnn/resnet50_tusimple.py --img images\
          --load_from model.pd --savedir ./vis
```

如果想要获取基于分割的车道线检测模型的分割结果，可以运行以下命令
```
# first you should add 'seg = True' in your config 
python tools/detect.py configs/scnn/resnet50_tusimple.py --img images\
          --load_from model.pd --savedir ./vis
```

### 4、测试模型检测速度
如果你想要测试模型的速度，你可以运行以下的命令。但是需要注意的是测试脚本使用python进行编写并未采用常见的C++，因此测试得到的模型检测速度会低于论文报告的结果，但是也可以用来衡量不同模型间检测速度快慢
```
 python tools/test_speed.py configs/condlane/cspresnet_50_culane.py --model_path output_dir/cspresnet_50_culane/model.pd
```

### 5、VisualDL可视化
如果你想可视化中间过程的loss，请在终端运行以下命令，其中log为存放日志的文件夹，更多的命令以及功能请参考[VisualDL](https://github.com/PaddlePaddle/VisualDL)
```
# 首先你需要在配置文件中加上use_visual = True，训练完后即可得到日志文件，将其放在log文件夹下
visualdl --logdir ./log
```

### 6、模型导出
如果你想将模型导出为预训练的格式(只保留模型权重去除优化器以及学习率的权重)，可以使用以下命令
```
python tools/train.py -c configs/ufld/mobilenetv3_culane.py --export output_dir/mobilenetv3_culane/epoch_51.pd
#如果模型权重中包含RepVGG模块，可以运行以下命令来将RepVGG中卷积进行重参数化。
#python tools/train.py -c config path --export model path --export_repvgg
```

## License
PPLanedet使用[MIT license](LICENSE)。但是我们仅允许您将PPLanedet用于学术用途。

## 致谢
* 非常感谢[PASSL](https://github.com/PaddlePaddle/PASSL)提供HOOK代码
* 非常感谢[lanedet](https://github.com/Turoad/lanedet)提供模型代码

## 引用
如果您认为我们的项目对您的研究有用，请引用我们的项目

```latex
@misc{PPlanedet,
    title={PPlanedet, A Toolkit for lane detection based on PaddlePaddle},
    author={Kunyang Zhou},
    howpublished = {\url{https://github.com/zkyseu/PPlanedet}},
    year={2022}
}
```

PPLanedet中复现的方法
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
