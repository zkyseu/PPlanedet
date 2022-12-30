model = dict(
    name='Detector',
)

backbone = dict(
    name='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    in_channels=[64, 128, 256],
    out_conv=True,
)

featuremap_out_channel = 128
featuremap_out_stride = 8

aggregator = dict(
    name='RESA',
    direction=['d', 'u', 'r', 'l'],
    alpha=2.0,
    iter=4,
    conv_stride=9,
)

sample_y=range(710, 150, -10)
heads = dict(
    name='LaneSeg',
    decoder=dict(name='BUSD'),
    thr=0.6,
    seg_loss = dict(name = 'CrossEntropyLoss',
                   weight = (0.4,1,1,1,1,1,1),
                   loss_weight = 1.),
    sample_y=sample_y,
)

epochs = 150
batch_size = 12
total_iter = (3616 // batch_size + 1) * epochs 

lr_scheduler = dict(
    name = 'PolynomialDecay',
    learning_rate = 0.025,
    decay_steps = total_iter
)

optimizer = dict(
  name = 'Momentum',
  weight_decay = 1e-4,
  momentum = 0.9
)       

img_height = 368
img_width = 640
cut_height = 160
ori_img_h = 720
ori_img_w = 1280

img_norm = dict(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

train_process = [
    dict(name='RandomRotation'),
    dict(name='RandomHorizontalFlip'),
    dict(name='Resize', size=(img_width, img_height)),
    dict(name='Normalize', img_norm=img_norm),
    dict(name='ToTensor'),
] 

val_process = [
    dict(name='Resize', size=(img_width, img_height)),
    dict(name='Normalize', img_norm=img_norm),
    dict(name='ToTensor', keys=['img']),
] 


dataset_path = '/home/fyj/zky/tusimple'
dataset = dict(
    train=dict(
        name='TuSimple',
        data_root=dataset_path,
        split='trainval',
        processes=train_process,
    ),
    val=dict(
        name='TuSimple',
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
    test=dict(
        name='TuSimple',
        data_root=dataset_path,
        split='test',
        processes=val_process,
    )
)

log_config = dict(
    name = 'LogHook',
    interval = 50
    )

custom_config = [dict(
    name = 'EvaluateHook'
    )]

device = 'gpu'
seed =  0
save_inference_dir = './inference'
output_dir = './output_dir'
best_dir = './output_dir/best_dir'
pred_save_dir = './pred_save'
num_workers = 4
num_classes = 6 + 1
view = False
ignore_label = 255
test_json_file='/home/fyj/zky/tusimple/test_label.json'
