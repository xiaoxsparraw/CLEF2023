import sys
sys.path.append('./CLIP')
import clip
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


train_meta_path = 'metadata/FungiCLEF2023_train_metadata_PRODUCTION.csv'
val_meta_path = 'metadata/FungiCLEF2023_val_metadata_PRODUCTION.csv'


train_df = pd.read_csv(train_meta_path)
val_df = pd.read_csv(val_meta_path)


train_code = set(train_df['level2Name'])
val_code = set(val_df['level2Name'])


union_code = train_code | val_code

code_lst = [str(elem) for elem in union_code]
code_lst.sort()
print(code_lst[:10])
model_arch = 'ViT-L/14@336px'

def extract_features(code_lst):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(model_arch, device=device)
    text = clip.tokenize(code_lst).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)

        print(text_features.shape)
        features = text_features.cpu().numpy()
        return features

features = extract_features(code_lst)
pca = PCA(n_components=0.99, svd_solver='full')
pca.fit(features)
pca_features = pca.transform(features)

code2idx = {code:i for i, code in enumerate(code_lst)}

code2feature_dict ={}
for key,value in zip(code_lst, pca_features):
    code2feature_dict[key] = value

np.save('meta_extract_features/code2feature.npy', code2feature_dict)
