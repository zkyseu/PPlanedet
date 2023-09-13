model = dict(
    name='Detector',
)

backbone = dict(
    name='CSPResNet',
    layers = [3,6,6,3],
    return_idx = [1,2,3],
    use_large_stem = True,
    depth_mult = 0.67,
    width_mult = 0.75,
    channels = [64, 128, 256, 512, 1024],
    use_alpha = True,
    out_conv = True,
    pretrain = "https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_m_pretrained.pdparams"
)

featuremap_out_channel = 512

griding_num = 100
num_classes = 6
heads = dict(name='LaneCls',
        dim = (griding_num + 1, 56, num_classes),
        loss = dict(name = 'SoftmaxFocalLoss',
                    gamma = 2),
        relation_loss = dict(name='ParsingRelationLoss'))

epochs = 150
batch_size = 13
total_iter = (3616 // batch_size + 1) * epochs 

lr_scheduler = dict(
    name = 'PolynomialDecay',
    learning_rate = 0.0003,
    decay_steps = total_iter
,
    power = 0.9
)

# optimizer = dict(
#   name = 'Momentum',
#   weight_decay = 1e-4,
#   momentum = 0.9
# )

optimizer = dict(
  name = 'AdamW',
)

ori_img_h = 720
ori_img_w = 1280
img_h = 288
img_w = 800
cut_height=0
sample_y = range(710, 150, -10)

img_norm = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

row_anchor = 'tusimple_row_anchor'

train_transform = [
    dict(name='RandomRotation', degree=(-6, 6)),
    dict(name='RandomUDoffsetLABEL', max_offset=100),
    dict(name='RandomLROffsetLABEL', max_offset=200),
    dict(name='GenerateLaneCls', row_anchor=row_anchor,
        num_cols=griding_num, num_classes=num_classes),
    dict(name='Resize', size=(img_w, img_h)),
    dict(name='Normalize', img_norm=img_norm),
    dict(name='ToTensor', keys=['img', 'cls_label']),
]

val_transform = [
    dict(name='Resize', size=(img_w, img_h)),
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
pred_save_dir = './pred_save'
num_workers = 4
num_classes = 6 + 1
view = False
ignore_label = 255
test_json_file='/home/fyj/zky/tusimple/test_label.json'
seg=False