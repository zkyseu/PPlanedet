## Training
In this documentation, we describe more details about training PPlanedet.

### 1. Train on single GPU
If you only have one gpu, you can run the following codes
```Shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/scnn/resnet50_tusimple.py # please change to your config path
```

### 2. Train on multi GPUs
If you have more than one gpu, you can run the following codes
```Shell
# training on 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch tools/train.py -c configs/scnn/resnet50_tusimple.py # please change to your config path
```

### 3. Loading pretrain weight
If you want to load the pretrain weight, you can run
```Shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/scnn/resnet50_tusimple.py --load pretrain weight path #
```

### 4. resume
If you want to resume the training program, you can run
```Shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/scnn/resnet50_tusimple.py --resume weight path #
```
