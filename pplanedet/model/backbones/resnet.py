from paddleseg.models.backbones.resnet_vd import ResNet_vd
from paddleseg.utils import utils
from paddle.vision.models.resnet import BottleneckBlock,get_weights_path_from_url
from ..builder import BACKBONES
import paddle.nn as nn
import paddle


model_urls = {
    'resnet18': ('https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams',
                 'cf548f46534aa3560945be4b95cd11c4'),
    'resnet34': ('https://paddle-hapi.bj.bcebos.com/models/resnet34.pdparams',
                 '8d2275cf8706028345f78ac0e1d31969'),
    'resnet50': ('https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams',
                 'ca6f485ee1ab0492d38f323885b0ad80'),
    'resnet101': ('https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams',
                  '02f35f034ca3858e1e54d4036443c92d'),
    'resnet152': ('https://paddle-hapi.bj.bcebos.com/models/resnet152.pdparams',
                  '7ad16a2f1e7333859ff986138630fd7a'),
} # if you need more models, you can refer to paddlevision



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)

@BACKBONES.register()
class ResNet(ResNet_vd):
    def __init__(self,
                 return_idx = [0,1,2],
                 pretrained = None,
                 out_conv=True,
                 cfg = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.return_idx = return_idx
        self.pretrained = pretrained
        self.out_convs = out_conv
        
        backbone_channels = [
            self.feat_channels[i] for i in return_idx
        ]
    
        if out_conv:
            self.out_conv = conv1x1(backbone_channels[-1],cfg.featuremap_out_channel) 
        
        self.init_weight() 

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        self.conv1_logit = y.clone()
        y = self.pool2d_max(y)

        # A feature list saves the output feature map of each stage.
        feat_list = []
        for idx,stage in enumerate(self.stage_list):
            if idx in self.return_idx:
                for block in stage:
                    y = block(y)
                feat_list.append(y)

        if self.out_convs:
            feat_list[-1] = self.out_conv(feat_list[-1]              )

        return feat_list    


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        # if dilation > 1:
        #     raise NotImplementedError(
        #         "Dilation > 1 not supported in BasicBlock")

        self.conv1 = nn.Conv2D(
            inplanes, planes, 3, stride=stride, bias_attr=False,dilation=dilation,padding=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes, 3,bias_attr=False,dilation=dilation,padding=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetvb(nn.Layer):
    def __init__(self,
                 depth=50,
                 width=64,
                 groups=1,
                 with_pool=True,
                 block = BasicBlock,
                 replace_stride_with_dilation = [],
                 in_channels=None,
                 **kwargs):
        super().__init__(**kwargs)

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self.groups = groups
        self.base_width = width
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2D(3,
                               self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, in_channels[0], layers[0])
        self.layer2 = self._make_layer(block, in_channels[1], layers[1], stride=2,dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, in_channels[2], layers[2], stride=2,dilate=replace_stride_with_dilation[1])
        if len(in_channels) > 3:
            if in_channels[3] > 0:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.expansion = block.expansion

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes,
                          planes * block.expansion,
                          1,
                          stride=stride,
                          bias_attr=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out_layers = [] 
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if not hasattr(self, name):
                continue
            layer = getattr(self, name)
            x = layer(x)
            out_layers.append(x)

        return out_layers 

@BACKBONES.register()
class ResNetWrapper(nn.Layer):

    def __init__(self, 
                resnet = '18',
                pretrained=True,
                replace_stride_with_dilation=[False, False, False],
                out_conv=False,
                fea_stride=8,
                out_channel=128,
                in_channels=[64, 128, 256, 512],
                cfg=None):
        super(ResNetWrapper, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels 

        self.model = eval(resnet)(
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation, in_channels=self.in_channels)
        self.out = None
        if out_conv:
            out_channel = 512
            for chan in reversed(self.in_channels):
                if chan < 0: continue
                out_channel = chan
                break
            self.out = conv1x1(
                out_channel * self.model.expansion, cfg.featuremap_out_channel)

    def forward(self, x):
        x = self.model(x)
        if self.out:
            x[-1] = self.out(x[-1])
        return x

def _resnet(arch, Block, depth, pretrained, **kwargs):
    model = ResNetvb(block = Block, depth = depth, **kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])

        param = paddle.load(weight_path)
        model.set_dict(param)

    return model

def resnet18(pretrained=False, **kwargs):
    return _resnet('resnet18', BasicBlock, 18, pretrained, **kwargs)

def resnet34(pretrained=False, **kwargs):
    return _resnet('resnet34', BasicBlock, 34, pretrained, **kwargs)

def resnet50(pretrained=False, **kwargs):
    return _resnet('resnet50', BottleneckBlock, 50, pretrained, **kwargs)
