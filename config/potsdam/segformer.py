from torch.utils.data import DataLoader

from geoseg.losses import *
from geoseg.datasets.potsdam_dataset import *
from geoseg.models.Segformer import Segformer
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from functools import partial


# training hparam
max_epoch = 200
ignore_index = len(CLASSES)
train_batch_size = 10
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

test_time_aug = 'd4'
output_mask_dir, output_mask_rgb_dir = None, None
weights_name = "rest_base_2-r18-768crop-ms-e45"
weights_path = "/data2/wangyuji/Geoseg/model_weights/potsdam/{}".format(weights_name)
test_weights_name = "rest_base_2-r18-768crop-ms-e45"
log_name = 'potsdam/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
gpus = [1]
strategy = None
pretrained_ckpt_path = None
# resume_ckpt_path = r'/home/featurize/work/pro_final/pro_final/model_weights/vaihingen/esegformer_base-r18-768crop-ms-e45/last.ckpt'
resume_ckpt_path = None
#  define the network
# net = Segformer(img_size=512, patch_size=4, in_chans=3, num_classes=num_classes, embed_dims=[ 40,80, 160,320 ],
#                  num_heads=[4, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
#                  attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
#                  depths=[3,4,6,4], sr_ratios=[8, 4, 2, 1])
net = Segformer(img_size=512, patch_size=4, in_chans=3, num_classes=num_classes, embed_dims=[ 40, 80, 200 ,320 ],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3,4,6,4], sr_ratios=[8, 4, 2, 1])

# define the loss
# loss = ESegformerLoss(ignore_index=ignore_index)
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
use_aux_loss = False

# define the dataloader

train_dataset = PotsdamDataset(data_root=r'/data2/wangyuji/Geoseg/potsdam/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug)

val_dataset =  PotsdamDataset(transform=val_aug)
test_dataset =  PotsdamDataset(data_root=r'/data2/wangyuji/Geoseg/potsdam/test',
                              transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          # shuffle=False,
                          drop_last=True)


val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)