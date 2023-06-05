import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import random
import time
import sys
sys.path.append( '/data1/PycharmProjects/FGVC10/pytorch-image-models')
import timm

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
from sklearn import metrics
from tqdm import tqdm

CFG = {
    'seed': 42,
    'resize_size': 460,
    'img_size': 448,
    'handle_openset': 'f1',
    'openset_ratio': 0.05,
    'num_classes': 1604,
    'model_arch': 'volo_d4_448',
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'valid_bs': 256*4,
    'checkpoints': [
        'chechkpoints/new_volo_d4_448.pth'
    ],
    'num_workers': 32 ,
    'device': 'cuda',
    'tta': 5
}



root_path='/data1/dataset/FungiCLEF2023/DF21/'
test_df = pd.read_csv('./metadata/FungiCLEF2023_public_test_metadata_PRODUCTION.csv')

valid_df = pd.read_csv('./metadata/FungiCLEF2023_val_metadata_PRODUCTION.csv')
train_df = pd.read_csv('./metadata/FungiCLEF2023_train_metadata_PRODUCTION.csv')

known_locality = set(train_df['locality'])
valid_known_idx = valid_df['locality'].isin(known_locality)
valid_unknown_idx = (valid_known_idx==False)
test_known_idx = test_df['locality'].isin(known_locality)
test_unknown_idx = (test_known_idx==False)

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


def get_model_with_checkpoint(checkpoint, device=torch.device(CFG['device'])):
    model = timm.create_model(CFG['model_arch'], num_classes=CFG['num_classes'], pretrained=False)
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    return model


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
    def __init__(self, df, transforms=get_inference_transforms_last(), root=root_path):
        self.root = root
        self.df = df
        self.transforms = transforms

    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):
        observation_id = self.df['observationID'][index]
        img = get_img(self.root + self.df['image_path'][index])
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        return img, observation_id


def inference(model, dataloader, device=CFG['device']):
    observation_id_lst = []
    probs_lst = []
    model.eval()
    with torch.no_grad():
        for imgs, observation_ids in tqdm(dataloader):
            imgs = imgs.to(device)
            outs = model(imgs)
            probs = F.softmax(outs)
            for observation_id, prob in zip(observation_ids, probs):
                observation_id_lst.append(observation_id.item())
                probs_lst.append(prob.cpu().numpy())
    probs_lst = np.array(probs_lst)
    return probs_lst, observation_id_lst


def tta_inference(model, dataloader, tta=CFG['tta']):
    probs_lst = []
    observation_id_lst = []

    for i in range(tta):
        if i == 0:
            probs_lst, observation_id_lst = inference(model, dataloader)
        else:
            t_probs_lst, _ = inference(model, dataloader)
            for j in range(len(t_probs_lst)):
                probs_lst[j] += t_probs_lst[j]

    probs_lst = probs_lst / tta

    return probs_lst, observation_id_lst



def average_probs_by_obs_id(probs, obs_id_lst):
    id2count = {}
    id2probs = {}
    for obs_id, probs in zip(obs_id_lst, probs):
        if obs_id not in id2count.keys():
            id2count[obs_id] = 1
            id2probs[obs_id] = probs
        else:
            id2count[obs_id] += 1
            id2probs[obs_id] += probs
    unique_obs_id_lst = []
    unique_probs_lst = []
    for obs_id, probs in id2probs.items():
        unique_obs_id_lst.append(obs_id)
        avg_probs = id2probs[obs_id] / id2count[obs_id]
        unique_probs_lst.append(avg_probs)
    return unique_obs_id_lst, unique_probs_lst



def get_predict_labels_and_max_probs(probs_lst):
    predict_labels = []
    max_probs_lst = []
    for prob in probs_lst:
        label = np.argmax(prob)
        max_probs_lst.append(prob[label])
        predict_labels.append(label)
    return np.array(predict_labels), np.array(max_probs_lst)


def get_threshold_in_valid(probs_lst, labels):
    threshold = 1 / CFG['num_classes']
    step = threshold
    predict_labels, max_probs_lst = get_predict_labels_and_max_probs(probs_lst)
    best_threshold = threshold
    best_f1_threshold = threshold
    best_acc = 0.0
    best_f1 = 0.0
    while threshold < 1.0:
        temp_labels = np.copy(predict_labels)
        temp_labels[max_probs_lst < threshold] = -1
        assert np.all((temp_labels == -1) == (max_probs_lst < threshold)), 'wrong -1 assign'
        acc = (temp_labels == labels).mean()
        f1 = metrics.f1_score(labels, temp_labels, average='macro')
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = threshold
        threshold += step
    print('best acc in valid: ', best_acc)
    print('best marcro-f1 in valid: ', best_f1)
    return best_threshold, best_acc, best_f1_threshold, best_f1


def handle_openset(class_id_lst, prob_lst, prob_threshold=None):
    handled_predict_lst = []
    sorted_idx = np.argsort(prob_lst)
    if prob_threshold is None:
        prob_threshold = prob_lst[sorted_idx[int(len(sorted_idx) * CFG['openset_ratio'])]]
    for class_id, prob in zip(class_id_lst, prob_lst):
        predict_label = class_id
        if prob < prob_threshold:
            predict_label = -1
        handled_predict_lst.append(predict_label)
    return handled_predict_lst


def unique_observation_id(observation_id_lst, array, prob_threshold):
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
    unique_observation_id_lst, class_id_lst, prob_lst = [], [], []
    softmaxscore = []
    for obs_id, probs in id2probs.items():
        unique_observation_id_lst.append(obs_id)
        label = np.argmax(probs)
        class_id_lst.append(label)
        prob_lst.append(probs[label])
        softmaxscore.append(probs)
    if CFG['handle_openset'] != 'none':
        class_id_lst = handle_openset(class_id_lst, prob_lst, prob_threshold)
    return unique_observation_id_lst, class_id_lst,softmaxscore




def write_csv(unique_observation_id_lst, class_id_lst ,softmxscore, file_name, file_name_with_prob = None):

    if file_name_with_prob is not None:
        with open(file_name_with_prob, 'w') as out:
            out.write('observation_id,class_id\n')
            for observation_id, label ,score in zip(unique_observation_id_lst, class_id_lst,softmxscore):
                out.write(str(observation_id) + ',' + str(label))
                out.writelines([','+str(i) for i in score ])
                out.write('\n')

    with open(file_name, 'w') as out:
        out.write('observation_id,class_id\n')
        for observation_id, label in zip(unique_observation_id_lst, class_id_lst):
            out.write(str(observation_id) + ',' + str(label) + '\n')




if __name__ == '__main__':
    assert CFG['handle_openset'] in ['f1', 'acc', 'ratio', 'none'], f'Not supported handle openset method:{CFG["handle_openset"]}'
    meta_feature_dim = 128
    best_f1_threshold = [0.27,0.23]
    for checkpoint in CFG['checkpoints']:
        seed_everything(CFG['seed'])

        print('checkpoint : ', checkpoint)
        device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
        model = get_model_with_checkpoint(checkpoint)
        unique_observation_id_lst, class_id_lst ,softmxscore = [],[],[]
        for i,test_idx in enumerate([test_known_idx,test_unknown_idx]):
            current_test_df = test_df[test_idx]
            current_test_df.set_index(np.arange(len(current_test_df)),inplace=True)
            dataset = InferenceDataset(current_test_df, get_inference_transforms_last())
            dataloader = DataLoader(
                dataset,
                batch_size=CFG['valid_bs'],
                num_workers=CFG['num_workers'],
                shuffle=False,
                pin_memory=False,
            )

            array = []

            probs_lst, observation_id_lst = tta_inference(model, dataloader)

            array.append(probs_lst)


            threshold = None
            if CFG['handle_openset'] == 'f1':
                threshold = best_f1_threshold[i]


            unique_observation_id_lst_part, class_id_lst_part ,softmxscore_part = unique_observation_id(observation_id_lst, array, threshold)
            unique_observation_id_lst.extend(unique_observation_id_lst_part)
            class_id_lst.extend(class_id_lst_part)
            softmxscore.extend(softmxscore_part)

    if CFG['handle_openset'] in ['f1', 'acc']:
        write_file_name = './inference/' +  'user_submission.csv'

    write_csv(unique_observation_id_lst, class_id_lst ,softmxscore,write_file_name)
