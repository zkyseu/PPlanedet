from paddleseg.models.backbones.resnet_vd import ResNet_vd
from paddleseg.cvlibs import manager

class ResNet(ResNet_vd):
    def __init__(self,return_idx = [0,1,2],**kwargs):
        super().__init__(**kwargs)
        self.return_idx = return_idx
    
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

        return feat_list    

@manager.BACKBONES.add_component
def ResNet50_vds(**args):
    model = ResNet(layers=50, **args)
    return model
