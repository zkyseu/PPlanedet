import paddle
import paddle.nn as nn
import paddle.nn.functional as F


from ..builder import BACKBONES

class non_bottleneck_1d(paddle.nn.Layer):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        self.conv3x1_1 = paddle.nn.Conv2D(in_channels=chann, out_channels=chann, kernel_size=(3, 1), stride=1, padding=(1, 0), bias_attr=True)
        self.conv1x3_1 = paddle.nn.Conv2D(in_channels=chann, out_channels=chann, kernel_size=(1, 3), stride=1, padding=(0, 1), bias_attr=True)
        self.bn1 = paddle.nn.BatchNorm(chann, epsilon=1e-03)
        self.conv3x1_2 = paddle.nn.Conv2D(in_channels=chann, out_channels=chann, kernel_size=(3, 1), stride=1, padding=(1 * dilated, 0), bias_attr=True,
                                              dilation=(dilated, 1))
        self.conv1x3_2 = paddle.nn.Conv2D(in_channels=chann, out_channels=chann, kernel_size=(1, 3), stride=1, padding=(0, 1 * dilated), bias_attr=True,
                                              dilation=(1, dilated))
        self.bn2 = paddle.nn.BatchNorm(chann, epsilon=1e-03)
        self.dropout = paddle.nn.Dropout(dropprob)
        self.p = dropprob

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = paddle.nn.functional.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = paddle.nn.functional.relu(output)
        output = self.conv3x1_2(output)
        output = paddle.nn.functional.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        if self.p != 0:
            output = self.dropout(output)
        return paddle.nn.functional.relu(output + input)

class DownsamplerBlock(paddle.nn.Layer):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = paddle.nn.Conv2D(in_channels=ninput, out_channels=noutput-ninput, kernel_size=3,
                                     stride=2, padding=1, bias_attr=True)
        self.pool = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.bn = paddle.nn.BatchNorm(noutput, epsilon=1e-3)

    def forward(self, input):
        output = paddle.concat(x=[self.conv(input), self.pool(input)], axis=1)
        output = self.bn(output)
        return paddle.nn.functional.relu(output)

class Encoder(paddle.nn.Layer):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)
        self.layers = paddle.nn.LayerList()
        self.layers.append(DownsamplerBlock(16, 64))
        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.1, 1))
        self.layers.append(DownsamplerBlock(64, 128))
        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))
        # only for encoder mode:
        self.output_conv = paddle.nn.Conv2D(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias_attr=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)
        for layer in self.layers:
            output = layer(output)
        if predict:
            output = self.output_conv(output)
        return output

class UpsamplerBlock(paddle.nn.Layer):
    def __init__(self, ninput, noutput, output_size=[16, 16]):
        super().__init__()
        self.conv = paddle.nn.Conv2DTranspose(ninput, noutput, kernel_size=3, stride=2, padding=1, bias_attr=True)
        self.bn = paddle.nn.BatchNorm(noutput, epsilon=1e-3)
        self.output_size = output_size

    def forward(self, input):
        output = self.conv(input, output_size=self.output_size)
        output = self.bn(output)
        return paddle.nn.functional.relu(output)

class Decoder(paddle.nn.Layer):
    def __init__(self, num_classes, raw_size=[576, 1640]):
        super().__init__()
        self.layers = paddle.nn.LayerList()
        self.raw_size = raw_size
        self.layers.append(UpsamplerBlock(128, 64, output_size=[raw_size[0] // 4, raw_size[1] // 4]))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(UpsamplerBlock(64, 16, output_size=[raw_size[0] // 2, raw_size[1] // 2]))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.output_conv = paddle.nn.Conv2DTranspose(16, num_classes, kernel_size=2, stride=2, padding=0, bias_attr=True)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        output = self.output_conv(output, output_size=[self.raw_size[0], self.raw_size[1]])

        return output

class Lane_exist(paddle.nn.Layer):
    def __init__(self, cfg, num_output):
        super().__init__()

        self.layers = nn.LayerList()

        self.layers.append(nn.Conv2D(128, 32, (3, 3), stride=1, padding=(4, 4), bias_attr=False, dilation=(4, 4)))
        self.layers.append(nn.BatchNorm2D(32, epsilon=1e-03))

        self.layers_final = nn.LayerList()

        self.layers_final.append(nn.Dropout2D(0.1))
        self.layers_final.append(nn.Conv2D(32, 5, (1, 1), stride=1, padding=(0, 0), bias_attr=True))

        self.maxpool = nn.MaxPool2D(2, stride=2)
        self.linear_dim = int(cfg.img_width / 16 * cfg.img_height / 16 * 5)
        self.linear1 = nn.Linear(self.linear_dim, 128)
        self.linear2 = nn.Linear(128, num_output)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = F.relu(output)

        for layer in self.layers_final:
            output = layer(output)

        output = F.softmax(output, axis=1)
        output = self.maxpool(output)
        output = output.reshape((-1, self.linear_dim))
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)

        return output


@BACKBONES.register()
class ERFNet(paddle.nn.Layer):
    def __init__(self, num_classes, use_exist = False,cfg = None):
        super().__init__()
        if use_exist:
            self.exist = Lane_exist(cfg,num_output=cfg.num_classes-1)
        self.encoder = Encoder(num_classes)
        raw_size = (cfg.img_height,cfg.img_width)
        self.decoder = Decoder(num_classes, raw_size=raw_size)

    def forward(self, input,**kwargs):
        outputs =  {}
        output = self.encoder(input)
        if hasattr(self,'exist'):
            outputs.update({'exist':self.exist(output)})
        outputs.update({'seg':self.decoder(output)})
        return outputs
