import sys

package_paths = [
    './pytorch-image-models',
    './CLIP'
]
for pth in package_paths:
    sys.path.append(pth)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import timm
import clip
from sklearn.decomposition import PCA
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import cv2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize, SmallestMaxSize
)

from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

CFG = {
    'seed': 42,
    'resize_size': 520,
    'img_size': 512,

    'num_classes': 1784,
    'endemic': False,

    'model_arch': 'convnextv2_large.fcmae_ft_in22k_in1k_384',

    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],

    'valid_bs': 256,
    'head': '2fc',
    'hidden_size': 2300,
    'dropout': 0.6,
    'metaBN': True,
    'checkpoints': [
        'checkpoints/pyramid_meta_seesawloss_convnextv2_large.fcmae_ft_in22k_in1k_384.pth'
    ],
    'prior_checkpoint': 'checkpoints/balanced_prior_best.pth',

    'num_workers': 4*4 ,
    'device': 'cuda',
    'tta': 1
}

predict_meta_path = 'metadata/SnakeCLEF2023-predict.csv'
root_path = '/data1/dataset/SnakeCLEF2023/'

train_meta_path = 'metadata/train_full.csv'
val_meta_path = 'metadata/SnakeCLEF2023-ValMetadata.csv'
test_meta_path = 'metadata/SnakeCLEF2023-PubTestMetadata.csv'

is_venomous_df = pd.read_csv('metadata/venomous_status_list.csv')

venomous_class = set(is_venomous_df['class_id'][is_venomous_df['MIVS'] == 1])


train_df = pd.read_csv(train_meta_path)
val_df = pd.read_csv(val_meta_path)
test_df = pd.read_csv(test_meta_path)
predict_df = pd.read_csv(predict_meta_path)


'''begin to extra  meta_feature  '''
code2feature = np.load('meta_extract_features/code2feature.npy',allow_pickle=True).item()

def get_pred_meta_features(df):
    meta_features = []
    for code in df['code']:
        if code in code2feature.keys():
            meta_features.append(code2feature[code])
        else:
            meta_features.append(code2feature['unknown'])
    meta_features = np.array(meta_features)
    return meta_features

'''  exrtact meta_feature end'''


''' prior  model    '''

class priormodel (nn.Module):
    def __init__(self,meta_feature_dim,classes_num):
        super().__init__()
        self.metaBN = nn.BatchNorm1d(meta_feature_dim)
        self.fc1 = nn.Linear(meta_feature_dim,512)
        self.fc2 = nn.Linear(512,256)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256,classes_num)
    def forward(self,x):
        x= self.metaBN(x)
        x = F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

''' images prediction'''


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_inference_transforms_last():
    return Compose([
        RandomResizedCrop(CFG['img_size'], CFG['img_size'], interpolation=cv2.INTER_CUBIC, scale=(0.6, 1.2)),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        Normalize(mean=CFG['mean'], std=CFG['std'], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


class InferenceDataset(Dataset):
    def __init__(self,meta_features, transforms=get_inference_transforms_last(),root=root_path):
        self.root = root
        self.predict_df = predict_df
        self.transforms = transforms
        self.meta_features = meta_features
    def __len__(self):

        return len(self.predict_df)

    def __getitem__(self, index):
        observation_id = self.predict_df['observation_id'][index]
        img = get_img(self.root + self.predict_df['image_path'][index])
        meta_feature = self.meta_features[index]
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        return img,meta_feature, observation_id


def inference(model, dataloader, device=CFG['device']):
    observation_id_lst = []
    probs_lst = []
    model.eval()
    with torch.no_grad():
        for imgs, meta_features,observation_ids in tqdm(dataloader):
            imgs = imgs.to(device).float()
            meta_features = meta_features.to(device).float()
            outs = model(imgs,meta_features)
            probs = F.softmax(outs)
            for observation_id, prob in zip(observation_ids, probs):
                observation_id_lst.append(observation_id.item())
                probs_lst.append(prob.cpu().numpy())
    probs_lst = np.array(probs_lst)
    return probs_lst, observation_id_lst



def unique_observation_id(observation_id_lst, array):
    id2count = {}
    id2probs = {}

    for i, obs_id in enumerate(observation_id_lst):
        sum_probs = sum([lst[i] for lst in array])
        avg_obs_id_probs = sum_probs / len(array)
        if obs_id in id2count.keys():

            id2count[obs_id] += 1
            id2probs[obs_id] += avg_obs_id_probs
        else:
            id2count[obs_id] = 1
            id2probs[obs_id] = avg_obs_id_probs
    for obs_id, count in id2count.items():
        id2probs[obs_id] = id2probs[obs_id] / count
    unique_observation_id_lst, class_id_lst = [], []
    softmaxscore = []
    for obs_id, probs in id2probs.items():
        unique_observation_id_lst.append(obs_id)
        label = np.argmax(probs)
        class_id_lst.append(label)
        softmaxscore.append(probs)
    return unique_observation_id_lst, class_id_lst,softmaxscore


def get_classid2endemic():
    train_df = pd.read_csv('./metadata/train_full.csv')
    class_id2endemic = {}
    for i in range(len(train_df)):
        class_id = train_df['class_id'][i]
        endemic = train_df['endemic'][i]
        class_id2endemic[class_id] = endemic
    return class_id2endemic


def unique_observation_id_endemic(observation_id_lst, array):
    obs_id2endemic = {}
    class_id2endemic = get_classid2endemic()
    for i in range(len(predict_df)):
        obs_id = predict_df['observation_id'][i]
        endemic = predict_df['endemic'][i]
        obs_id2endemic[obs_id] = endemic
    id2count = {}
    id2probs = {}
    for i, obs_id in enumerate(observation_id_lst):
        sum_probs = sum([lst[i] for lst in array])
        avg_obs_id_probs = sum_probs / len(array)
        if obs_id in id2count.keys():

            id2count[obs_id] += 1
            id2probs[obs_id] += avg_obs_id_probs
        else:
            id2count[obs_id] = 1
            id2probs[obs_id] = avg_obs_id_probs
    for obs_id, count in id2count.items():
        id2probs[obs_id] = id2probs[obs_id] / count
    unique_observation_id_lst, class_id_lst = [], []
    for obs_id, probs in id2probs.items():
        unique_observation_id_lst.append(obs_id)
        sort_idx = np.argsort(probs)[::-1]
        label = None
        for idx in sort_idx:
            label = idx
            if obs_id2endemic[obs_id] == class_id2endemic[label]:
                break
            else:
                print('endemic works in obs_id: ', obs_id)
        class_id_lst.append(label)
    return unique_observation_id_lst, class_id_lst


def write_csv(observation_id_lst, array, file_name,meta_feature_dim,with_prob_filename=None):
    if CFG['endemic']:
        unique_observation_id_lst, class_id_lst = unique_observation_id_endemic(observation_id_lst, array)
        with open(file_name, 'w') as out:
            out.write('observation_id,class_id\n')
            for observation_id, label in zip(unique_observation_id_lst, class_id_lst):
                out.write(str(observation_id) + ',' + str(label) + '\n')

    else:
        unique_observation_id_lst, class_id_lst , softmaxscore= unique_observation_id(observation_id_lst, array)

        model = priormodel(meta_feature_dim,CFG['num_classes'])
        model = nn.DataParallel(model)
        model.to(device)
        checkpoint = CFG['prior_checkpoint']
        model.load_state_dict(torch.load(checkpoint),strict=True)


        model.eval()
        threhold = 1.1
        count = 0
        count1 =0
        max_prior_and_probs = []
        with torch.no_grad():
            for i,prop in enumerate(softmaxscore):
                if np.max(prop)<threhold:
                    code = ''
                    for obid,code2 in zip(predict_df['observation_id'],predict_df['code']):
                        if obid == unique_observation_id_lst[i]:
                            code = code2
                            break
                    if code in code2feature.keys():
                        meta_feature = code2feature[code]
                    else:
                        meta_feature = code2feature['unknown']
                    out = model(torch.tensor(meta_feature).unsqueeze(0).float().to(device))

                    prior = F.softmax(out,dim = -1).cpu().numpy()

                    prop = prop * prior
                    new_pred = np.argmax(prop)

                    max_prior_and_probs.append(prop[0][new_pred].item())
                    if new_pred != class_id_lst[i]:
                        count += 1
                        class_id_lst[i] = new_pred
            print('prior process count:',count)

        '''
        prior process end
        '''

        '''start venomous process'''

        sorted_idx = np.argsort(max_prior_and_probs)
        low_credict_num = input(len(predict_df)*0.06)
        for j in sorted_idx[0:low_credict_num]:
            top = torch.topk(torch.tensor(softmaxscore[j]),5)
            for idx in top.indices:

                if idx.item() in venomous_class:
                    class_id_lst[j] = idx.item()
                    count1 += 1
                    break
        print('venomous process count:',count1)

        if with_prob_filename is not None:
            with open(with_prob_filename, 'w') as out:
                out.write('observation_id,class_id')
                out.writelines([','+str(i) for i in range(CFG['num_classes'])])
                out.write('\n')
                for observation_id, label , score in zip(unique_observation_id_lst, class_id_lst,softmaxscore):
                    out.write(str(observation_id) + ',' + str(label))
                    out.writelines([','+str(i) for i in score ])
                    out.write('\n')

        with open(file_name, 'w') as out:
            out.write('observation_id,class_id\n')
            for observation_id, label in zip(unique_observation_id_lst, class_id_lst):
                out.write(str(observation_id) + ',' + str(label) + '\n')

class metamodel(nn.Module):
    def __init__(self,  model_arch, feature_dim, meta_feature_dim, num_classes) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_arch, num_classes=0, pretrained=False)
        if CFG['metaBN']:
            self.meta_batchnorm = nn.BatchNorm1d(meta_feature_dim)

        if CFG['dropout'] > 0:
            self.dropout = nn.Dropout(CFG['dropout'])

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
        if CFG['dropout'] > 0:
            outs = self.dropout(outs)
        if CFG['metaBN']:
            meta_feature = self.meta_batchnorm(meta_feature)
        features = torch.cat((outs, meta_feature,mid_feature), dim=-1)
        outs = self.head(features)
        return outs


if __name__ == '__main__':

    for checkpoint in CFG['checkpoints']:
        seed_everything(CFG['seed'])

        print('checkpoint : ', checkpoint)

        meta_features = get_pred_meta_features(predict_df)

        meta_feature_dim = meta_features.shape[1]
        dataset = InferenceDataset(meta_features)
        dataloader = DataLoader(
            dataset,
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        device = torch.device(CFG['device'])

        temp_model = timm.create_model(CFG['model_arch'], num_classes=0, pretrained=False)
        feature_dim = temp_model(torch.rand((1, 3, CFG['img_size'], CFG['img_size'])))[0].shape[1]
        del temp_model
        model = metamodel(CFG['model_arch'], feature_dim, meta_feature_dim, CFG['num_classes'])
        model = nn.DataParallel(model)
        model.to(device)
        array = []

        model.load_state_dict(torch.load(
            checkpoint
        ), strict=True)

        probs_lst = []
        observation_id_lst = []

        for i in range(CFG['tta']):
            if i == 0:
                probs_lst, observation_id_lst = inference(model, dataloader)
            else:
                t_probs_lst, t_observation_id_lst = inference(model, dataloader)
                for j in range(len(t_probs_lst)):
                    probs_lst[j] += t_probs_lst[j]

        probs_lst = probs_lst / CFG['tta']

        array.append(probs_lst)

        del model

        write_file_name = './inference/' +  'self_user_submission.csv'

        write_csv(observation_id_lst, array, write_file_name,meta_feature_dim)

