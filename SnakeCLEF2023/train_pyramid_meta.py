import sys
package_paths = [
    "/root/autodl-tmp/projects/pytorch-image-models",
]


for pth in package_paths:
    sys.path.append(pth)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
import cv2
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize, SmallestMaxSize
)
import time
import timm
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD, AdamW, RMSprop
from torch import nn
import random
from tqdm import tqdm
from PIL import Image
import numpy as np
import logging
from sklearn.model_selection import GroupKFold, StratifiedKFold
import pandas as pd


CFG = {
    'seed': 42,
    'model_arch': 'convnextv2_large.fcmae_ft_in22k_in1k_384',
    'patch': 16,
    'mean':[0.485, 0.456, 0.406] ,
    'std':[0.229, 0.224, 0.225],
    'head':'2fc' ,
    'hidden_size': 2300,
    'dropout': 0.6,
    'checkpoints': './checkpoints/seesawloss_convnextv2_large.fcmae_ft_in22k_in1k_512.pth',
    'mix_type': 'randommix',
    'mix_prob': 0.9,
    'img_size': 512,
    'class_num': 1784,
    'warmup_epochs': 1,
    'warmup_lr_factor': 0.01,
    'epochs': 15,
    'train_bs': 32,
    'valid_bs': 64,
    'lr': 1.5e-4/2,
    'min_lr': 1e-8,
    'differLR': False  ,
    'bacbone_lr_factor': 0.2,
    'num_workers': 16,
    'device': 'cuda',
    'smoothing': 0.1,
    'weight_decay': 2e-5,
    'accum_iter': 1,
    'verbose_step': 1,


}

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(f"logs/pyramid{CFG['model_arch']}_train.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


train_data_root = '/root/autodl-tmp/datasets/'
val_data_root = '/root/autodl-tmp/datasets/val/SnakeCLEF2023-large_size/'
train_df = pd.read_csv('./metadata/train_full.csv')

valid_df = pd.read_csv('./metadata/SnakeCLEF2023-ValMetadata.csv')
meta_path = '/root/autodl-tmp/projects/snake/meta_extract_features/'
train_meta_feature_path = meta_path+'clip_ViT-L_14_336px_train.npy'
val_meta_feature_path = meta_path+'clip_ViT-L_14_336px_val.npy'


class FGVCDataset(Dataset):
    def __init__(self, df, data_root,
                 meta_feature_path ,
                 transforms=None,
                 output_label=True,
                 one_hot_label=False
                 ):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.meta_feature_path = meta_feature_path
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        self.meta_features = np.load(self.meta_feature_path)

        if output_label == True:
            self.labels = self.df['class_id'].values

            if one_hot_label is True:
                self.labels = np.eye(self.df['class_id'].max() + 1)[self.labels]

    def __len__(self):

        return self.df.shape[0]

    def __getitem__(self, index: int):

        meta_feature = self.meta_features[index]
        if self.output_label:
            target = self.labels[index]

        image_path = self.data_root + self.df.loc[index]['image_path']
        img = get_img(image_path)

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.output_label == True:
            return img, meta_feature,target
        else:
            return img


def get_train_transforms():
    return Compose([
        RandomResizedCrop(CFG['img_size'], CFG['img_size'],
                          interpolation=cv2.INTER_CUBIC, scale=(0.5, 1.3)),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.3),
        PiecewiseAffine(p=0.5),
        RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0),
        OneOf([
            OpticalDistortion(distort_limit=1.0),
            GridDistortion(num_steps=5, distort_limit=1.),

        ], p=0.5),

        Normalize(mean=CFG['mean'], std=CFG['std'],
                  max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)



def get_valid_transforms():
    return Compose([
        Resize(CFG['img_size'], CFG['img_size'],
               interpolation=cv2.INTER_CUBIC),
        Normalize(mean=CFG['mean'], std=CFG['std'],
                  max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)



def prepare_dataloader(train_df, val_df, train_idx, val_idx):
    train_ = train_df.loc[train_idx, :].reset_index(drop=True)
    valid_ = val_df.loc[val_idx, :].reset_index(drop=True)

    train_ds = FGVCDataset(train_, train_data_root, transforms=get_train_transforms(),
                            meta_feature_path=train_meta_feature_path)
    valid_ds = FGVCDataset(valid_, val_data_root, transforms=get_valid_transforms(),
                            meta_feature_path=val_meta_feature_path)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=CFG['num_workers']
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader



def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()

    running_loss = None
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, meta_features,image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        meta_features = meta_features.to(device).float()
        with autocast():
            image_preds = model(imgs, meta_features)

            image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
            image_targets_all += [image_labels.detach().cpu().numpy()]
            loss = loss_fn(image_preds, image_labels)
            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    accuracy = (image_preds_all == image_targets_all).mean()
    print('Train multi-class accuracy = {:.4f}'.format(accuracy))
    logger.info(' Epoch: ' + str(epoch) + ' Train multi-class accuracy = {:.4f}'.format(accuracy))
    logger.info(' Epoch: ' + str(epoch) + ' Train loss = {:.4f}'.format(running_loss))

    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def generate_mask_random(imgs, patch=CFG['patch'], mask_token_num_start=14, lam=0.5):
    _, _, W, H = imgs.shape
    assert W % patch == 0
    assert H % patch == 0
    p = W // patch

    mask_ratio = 1 - lam
    num_masking_patches = min(p**2, int(mask_ratio * (p**2)) + mask_token_num_start)
    mask_idx = np.random.permutation(p**2)[:num_masking_patches]
    lam = 1 - num_masking_patches / (p**2)
    return mask_idx, lam


def get_mixed_data(imgs, image_labels, mix_type):
    mix_lst = ['cutmix', 'tokenmix', 'mixup',  'randommix']
    assert mix_type in mix_lst, f'Not Supported mix type: {mix_type}'
    if mix_type == 'randommix':
        mix_type = random.choice(mix_lst[:-2])

    if mix_type == 'mixup':
        alpha = 2.0
        rand_index = torch.randperm(imgs.size()[0]).cuda()
        target_a = image_labels
        target_b = image_labels[rand_index]
        lam = np.random.beta(alpha, alpha)
        imgs = imgs * lam + imgs[rand_index] * (1 - lam)
    elif mix_type == 'cutmix':
        beta = 1.0
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(imgs.size()[0]).cuda()
        target_a = image_labels
        target_b = image_labels[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
    elif mix_type == 'tokenmix':
        B, C, W, H = imgs.shape
        mask_idx, lam = generate_mask_random(imgs)
        rand_index = torch.randperm(imgs.size()[0]).cuda()
        p = W // CFG['patch']
        patch_w = CFG['patch']
        patch_h = CFG['patch']
        for idx in mask_idx:
            row_s = idx // p
            col_s = idx % p
            x1 = patch_w * row_s
            x2 = x1 + patch_w
            y1 = patch_h * col_s
            y2 = y1 + patch_h
            imgs[:, :, x1:x2, y1:y2] = imgs[rand_index, :, x1:x2, y1:y2]

        target_a = image_labels
        target_b = image_labels[rand_index]

    return imgs, target_a, target_b, lam


def train_one_epoch_mix(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False, mix_type=CFG['mix_type']):
    model.train()

    running_loss = None
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, meta_features,image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        meta_features = meta_features.to(device).float()

        if np.random.rand(1) < CFG['mix_prob']:
            imgs, target_a, target_b, lam = get_mixed_data(imgs, image_labels, mix_type)
            with autocast():
                image_preds = model(imgs,meta_features)
                loss = loss_fn(image_preds, target_a) * lam + loss_fn(image_preds, target_b) * (1. - lam)
                scaler.scale(loss).backward()
        else:
            with autocast():
                image_preds = model(imgs,meta_features)
                loss = loss_fn(image_preds, image_labels)
                scaler.scale(loss).backward()
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * .99 + loss.item() * .01
        if running_loss >10:
            print(epoch)
        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None and schd_batch_update:
                scheduler.step()

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
            description = f'epoch {epoch} loss: {running_loss:.4f}'
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    accuracy = (image_preds_all == image_targets_all).mean()

    print('Train multi-class accuracy = {:.4f}'.format(accuracy))
    logger.info(' Epoch: ' + str(epoch) + ' Train multi-class accuracy = {:.4f}'.format(accuracy))
    logger.info(' Epoch: ' + str(epoch) + ' Train loss = {:.4f}'.format(running_loss))

    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, meta_features,image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        meta_features = meta_features.to(device).float()
        image_preds = model(imgs,meta_features)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        openset_idx = image_labels == -1
        image_labels[openset_idx] = 0
        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum / sample_num:.4f}'
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)

    accuracy = (image_preds_all == image_targets_all).mean()
    print('validation multi-class accuracy = {:.4f}'.format(accuracy))
    logger.info(' Epoch: ' + str(epoch) + ' validation multi-class accuracy = {:.4f}'.format(accuracy))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()
    return accuracy


def get_loss_weight(df):
    df_class_id= np.array(df['class_id'])
    weight=np.zeros(CFG['class_num'], dtype=np.int32)
    for i in df_class_id:
        weight[i] += 1
    weight[weight == 0] = np.mean(weight)
    weight = 1.0 / weight
    return weight


class SeesawLossWithLogits(nn.Module):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the techinical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.
    Args:
    class_counts: The list which has number of samples for each class.
                  Should have same length as num_classes.
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 as a default by following the original paper.
    """

    def __init__(self, class_counts: np.array, p: float = 0.8):
        super().__init__()

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        falses = torch.ones(len(class_counts), len(class_counts))
        self.s = torch.where(conditions, trues, falses)

        self.eps = 1.0e-6

    def forward(self, logits, targets):
        targets = nn.functional.one_hot(targets,num_classes=CFG['class_num']).float().to(targets.device)
        self.s = self.s.to(targets.device)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = (
            (1 - targets)[:, None, :]
            * self.s[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()

class metamodel(nn.Module):
    def __init__(self,  model_arch, feature_dim, meta_feature_dim, num_classes) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_arch, num_classes=0, pretrained=True)

        if CFG['differLR'] and CFG['bacbone_lr_factor'] == 0:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if CFG['dropout']>0:
            self.dropout = nn.Dropout(CFG['dropout'])

        self.meta_batchnorm = nn.BatchNorm1d(meta_feature_dim)
        if CFG['head'] == '1fc':
            self.head = nn.Linear(feature_dim + meta_feature_dim, num_classes)
        elif CFG['head'] == '2fc':
            self.head = nn.Sequential(nn.Linear(feature_dim + meta_feature_dim+384, CFG['hidden_size']),
                                      nn.ReLU(),
                                      nn.Linear(CFG['hidden_size'], num_classes))
        else:
            assert False, f'head not found'
    def forward(self, x, meta_feature):
        outs,mid_feature = self.backbone(x)
        if CFG['dropout']>0:
            outs = self.dropout(outs)
        meta_feature = self.meta_batchnorm(meta_feature)
        features = torch.cat((outs, meta_feature,mid_feature), dim=-1)
        outs = self.head(features)
        return outs


if __name__ == '__main__':
    seed_everything(CFG['seed'])
    logger.info(CFG)

    trn_idx = np.arange(train_df.shape[0])
    val_idx = np.arange(valid_df.shape[0])

    df_class_id = np.array(train_df['class_id'])
    class_counts = np.bincount(df_class_id)

    temp_model = timm.create_model(CFG['model_arch'], num_classes=0, pretrained=False)
    feature_dim = temp_model(torch.rand((1, 3, CFG['img_size'], CFG['img_size'])))[0].shape[1]
    del temp_model
    meta_feature_dim = np.load(train_meta_feature_path).shape[1]
    print('feature_dim:', feature_dim, 'meta_feature_dim:', meta_feature_dim)

    device = torch.device(CFG['device'])
    model = metamodel(CFG['model_arch'], feature_dim, meta_feature_dim, CFG['class_num'])
    model = nn.DataParallel(model)
    model.to(device)

    train_loader, val_loader = prepare_dataloader(train_df, valid_df, trn_idx, val_idx)

    scaler = GradScaler()

    backbone_params = list(map(id, model.module.backbone.parameters()))
    head_params = filter(lambda p: id(p) not in backbone_params, model.parameters())
    lr_cfg = [ {'params': model.module.backbone.parameters(), 'lr': CFG['lr'] * CFG['bacbone_lr_factor']},
                {'params': head_params , 'lr': CFG['lr']}]

    if CFG['differLR']:
        if CFG['bacbone_lr_factor']>0:
            optimizer = torch.optim.AdamW(lr_cfg, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        else:
            optimizer = torch.optim.AdamW([{'params': head_params , 'lr': CFG['lr']}],
                                          lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])


    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['epochs'] - CFG['warmup_epochs'], eta_min=CFG['min_lr']
    )
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=CFG['warmup_lr_factor'], total_iters=CFG['warmup_epochs']
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[CFG['warmup_epochs']]
    )


    loss_tr = SeesawLossWithLogits(class_counts)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=CFG['smoothing']).to(device)

    best_acc = 0.0
    for epoch in range(CFG['epochs']):
        print(optimizer.param_groups[0]['lr'])

        if CFG['mix_type'] == 'none':
            train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler)
        else:
            train_one_epoch_mix(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler)

        temp_acc = 0.0
        with torch.no_grad():
            temp_acc = valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)
            if temp_acc > best_acc:
                torch.save(model.state_dict(), './checkpoints/pyramid_meta_seesawloss_{}_mixtype_{}_mixprob_{}_seed_{}_ls_{}_epochs_{}_differLR_{}_head_{}.pth'.format(
                                                CFG['model_arch'],
                                                CFG['mix_type'],
                                                CFG['mix_prob'],
                                                CFG['seed'],
                                                CFG['smoothing'],
                                                CFG['epochs'],
                                                CFG['differLR'],
                                                CFG['head']))
        if temp_acc > best_acc:
            best_acc = temp_acc

    del model, optimizer, train_loader, val_loader, scaler, scheduler
    print(best_acc)
    logger.info('BEST-Valid-ACC: ' + str(best_acc))
    torch.cuda.empty_cache()
