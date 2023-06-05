import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam,AdamW
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import random
import logging
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from imbalanced import ImbalancedDatasetSampler

CFG={
    'classes_num':1604,
    'train_bs': 512,
    'val_bs': 1024,
    'lr': 1e-5*3,
    'min_lr': 1e-5/10,
    'weight_decay': 2e-5,
    'epochs': 60,
    'warmup_epochs': 1,
    'warmup_lr_factor': 0.01,
    'accum_iter': 1,
    'verbose_step': 1,

}



logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(f"logs/train_prior.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


root_path ='./metadata/'

val_meta_path = 'meta_extract_features/clip_ViT-L_14_336px_val.npy'
train_meta_path = 'meta_extract_features/clip_ViT-L_14_336px_train.npy'

train_df_path = 'FungiCLEF2023_train_metadata_PRODUCTION.csv'
val_df_path = 'FungiCLEF2023_val_metadata_PRODUCTION.csv'



train_df = pd.read_csv(root_path+train_df_path)
val_df = pd.read_csv(root_path+val_df_path)



element = ['Habitat', 'MetaSubstrate']

code2feature_l2 = np.load('meta_extract_features/fungi_level2Name_code2feature.npy',allow_pickle=True).item()
code2feature = np.load('meta_extract_features/fungi_Habitat_MetaSubstrate_code2feature.npy',allow_pickle=True).item()

def get_pred_meta_features(df):
    meta_features = []
    for h,m in zip(df['Habitat'],df['MetaSubstrate']):
        code = str(h)+'.'+ str(m)
        if code in code2feature.keys():
            meta_features.append(code2feature[code])
        else:
            meta_features.append(code2feature['unknown.unknown'])
    meta_features = np.array(meta_features)
    return meta_features


def get_pred_meta_features_l2(df):
    meta_features = []
    for code in df['level2Name']:

        if code in code2feature_l2.keys():
            meta_features.append(code2feature_l2[code])
        else:
            meta_features.append(code2feature_l2['unknown'])
    meta_features = np.array(meta_features)
    return meta_features


class code2priorDataset(Dataset):
    def __init__(self,df,meta_features):
        super().__init__()
        self.df = df
        self.meta_features = meta_features
        self.labels = df['class_id'].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        label = torch.zeros(CFG['classes_num'])
        label = self.labels[idx]
        meta_feature = self.meta_features[idx]
        meta_feature = torch.tensor(meta_feature).float()

        return meta_feature,label
    def get_labels(self):
        return self.labels

class FCN(nn.Module):
    def __init__(self,dim=256):
        super().__init__()
        self.fc1 = nn.Linear(dim,dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim,dim)
        self.drop = nn.Dropout(0.3)

    def forward(self,x):
        o = self.relu(self.fc1(x))
        o = self.drop(o)
        o = self.relu(self.fc2(o))
        return o + x

class priormodel (nn.Module):
    def __init__(self,meta_feature_dim,classes_num):
        super().__init__()
        self.metaBN = nn.BatchNorm1d(meta_feature_dim)
        self.fc1 = nn.Linear(meta_feature_dim,256)
        self.fcn1 = FCN(dim=256)
        self.fcn2 = FCN(dim=256)
        self.fc3 = nn.Linear(256, classes_num)

    def forward(self,x):
        x= self.metaBN(x)
        x = F.relu(self.fc1(x))
        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fc3(x)

        return x

def get_dataloader(train_df,val_df,trainbs = CFG['train_bs'],valbs = CFG['val_bs']):
    train_meta_feature = get_pred_meta_features(train_df) + get_pred_meta_features_l2(train_df)
    val_meta_feature  = get_pred_meta_features(val_df) + get_pred_meta_features_l2(val_df)
    train_dataset = code2priorDataset(train_df,train_meta_feature)
    val_dataset = code2priorDataset(val_df,val_meta_feature)
    train_dataloader = DataLoader(train_dataset,batch_size=trainbs,
                                  drop_last=(int(len(train_dataset)%trainbs) == 1),num_workers=4,
                                  sampler=ImbalancedDatasetSampler(train_dataset))
    val_dataloader = DataLoader(val_dataset,batch_size=valbs,shuffle=False,num_workers=4)
    return train_dataloader,val_dataloader

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()

    running_loss = None
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        r = torch.randn_like(imgs).to(device).float()

        with autocast(enabled=False):
            image_preds = model(imgs)
            r_preds = model(r)
            image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
            image_targets_all += [image_labels.detach().cpu().numpy()]

            loss = loss_fn(image_preds, image_labels,r_preds)

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


def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    right_num_all = 0
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs,image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        r = torch.randn_like(imgs).to(device).float()

        image_preds = model(imgs)
        r_preds = model(r)
        scores = image_preds[torch.arange(image_preds.shape[0]),image_labels].detach().cpu().numpy()
        right_num = np.sum((scores >0))
        right_num_all += right_num
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        openset_idx = image_labels == -1
        image_labels[openset_idx] = 0
        loss = loss_fn(image_preds, image_labels,r_preds)
        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum / sample_num:.4f}'
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)

    accuracy = right_num_all / sample_num
    print('validation multi-class accuracy = {:.4f}'.format(accuracy))
    logger.info(' Epoch: ' + str(epoch) + ' validation multi-class accuracy = {:.4f}'.format(accuracy))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()
    return accuracy


def get_scheduler():
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['epochs'] - CFG['warmup_epochs'], eta_min=CFG['min_lr']
    )
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=CFG['warmup_lr_factor'], total_iters=CFG['warmup_epochs']
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[CFG['warmup_epochs']]
    )
    return scheduler

class priorloss(nn.Module):


    def __init__(self, alpha = 10):
        super().__init__()
        self.alpha = alpha


    def forward(self, inputs, targets,r):
        targets = F.one_hot(targets,num_classes=CFG['classes_num']).float().to(targets.device)
        weight = torch.where(targets == 1, self.alpha * torch.ones_like(targets), torch.ones_like(targets))
        loss_loc = F.binary_cross_entropy_with_logits(inputs, targets, weight)


        loss_r = F.binary_cross_entropy_with_logits(r,torch.zeros_like(targets))
        loss = loss_loc + loss_r
        return loss

if __name__ == '__main__':
    logger.info(CFG)
    device = torch.device('cuda')
    train_loader,val_loader = get_dataloader(train_df,val_df)
    meta_feature_dim = 128
    print(meta_feature_dim)
    print(len(element))
    model = priormodel(meta_feature_dim,CFG['classes_num'])
    model = nn.DataParallel(model)
    model.to(device)

    scaler = GradScaler()

    optimizer = AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])

    loss_tr = priorloss(CFG['classes_num'])
    scheduler = get_scheduler()
    best_acc = 0.0
    for epoch in range(CFG['epochs']):
        print(optimizer.param_groups[0]['lr'])

        train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler)

        temp_acc = 0.0

        with torch.no_grad():
            temp_acc = valid_one_epoch(epoch, model, loss_tr, val_loader, device, scheduler=None, schd_loss_update=False)
            if epoch == 60-1:
                torch.save(model.state_dict(), f'checkpoints/fungi_balanced_prior_h_m_l2'
                           +'.pth')

        if temp_acc > best_acc:
            best_acc = temp_acc

    del model, optimizer, train_loader, val_loader, scaler, scheduler
    print(best_acc)
    logger.info('BEST-Valid-loss: ' + str(best_acc))
    torch.cuda.empty_cache()

