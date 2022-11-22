# PPlanedet: A development Toolkit for lane detection based on PaddlePaddle

In this project, we develop a toolkit for lane detection to facilitate research. Especially, PPlanedet is built upon [Paddleseg](https://github.com/PaddlePaddle/PaddleSeg) which is a development Toolkit for segmentation based on PaddlePaddle.

If you do not have enough compute resource, we recommend that you can run our project at [AiStudio](https://aistudio.baidu.com/aistudio/index?ad-from=m-title), which can provide V100 with 32GB memory.

## News 
<ul class="nobull">
  <li>[2022-11-22] :fire: we release the project code. We now only reproduce the <a href="https://arxiv.org/pdf/1712.06080.pdf">SCNN</a> with 93.70% accuracy in Tusimple dataset. Pretrain model will be updated in the following days. We will also release the eval and demo code in the following days.

</ul>

## Introduction
PPlanedet is developed for lane detection based on PaddlPaddle, which is a high performance Deep learning framework. The idea behind the pplanedet is to facilitate researchers who use PaddlePaddle to do research about lane detection. If you have any suggestions about our project, you can contact me.

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
