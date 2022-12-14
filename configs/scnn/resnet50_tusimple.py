model = dict(
    name='Detector',
)

backbone = dict(
    name='ResNet',
    output_stride=8,
    multi_grid=[1, 1, 1],
    return_idx=[0,1,2],
    pretrained='https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz',
)

featuremap_out_channel = 128

aggregator = dict(
    name='SCNN',
)

sample_y=range(710, 150, -10)

heads = dict(
    name='LaneSeg',
    decoder=dict(name='PlainDecoder'),
    thr=0.6,
    seg_loss = dict(name = 'CrossEntropyLoss',
                   weight = (0.4,1,1,1,1,1,1),
                   loss_weight = 1.),
    sample_y=sample_y,
)

epochs = 100
batch_size = 10
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

train_transform = [
    dict(name='RandomRotation'),
    dict(name='RandomHorizontalFlip'),
    dict(name='Resize', size=(img_width, img_height)),
    dict(name='Normalize', img_norm=img_norm),
    dict(name='ToTensor'),
] 

val_transform = [
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
        processes=train_transform,
    ),
    val=dict(
        name='TuSimple',
        data_root=dataset_path,
        split='test',
        processes=val_transform,
    ),
    test=dict(
        name='TuSimple',
        data_root=dataset_path,
        split='test',
        processes=val_transform,
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
num_workers = 4
num_classes = 6 + 1
view = False
ignore_label = 255
test_json_file='/home/fyj/zky/tusimple/test_label.json'