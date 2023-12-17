# zeroshot validation reproduction for DigestPath
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import wandb
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score, matthews_corrcoef, accuracy_score, classification_report

from sklearn.linear_model import SGDClassifier
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


def eval_metrics(y_true, y_pred, y_pred_proba = None, average_method='weighted'):
    assert len(y_true) == len(y_pred)
    if y_pred_proba is None:
        auroc = np.nan
    elif len(np.unique(y_true)) > 2:
        print('Multiclass AUC is not currently available.')
        auroc = np.nan
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auroc = auc(fpr, tpr)
    f1 = f1_score(y_true, y_pred, average = average_method)
    print(classification_report(y_true, y_pred))
    precision = precision_score(y_true, y_pred, average = average_method)
    recall = recall_score(y_true, y_pred, average = average_method)
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    tp,fp,tn,fn = 0,0,0,0
    for i in range(len(y_pred)):
        if y_true[i]==y_pred[i]==1:
           tp += 1
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
           fp += 1
        if y_true[i]==y_pred[i]==0:
           tn += 1
        if y_pred[i]==0 and y_true[i]!=y_pred[i]:
           fn += 1
    if (tp+fn) == 0: sensitivity = np.nan
    else: sensitivity = tp/(tp+fn) # recall
    if (tn+fp) == 0: specificity = np.nan
    else: specificity = tn/(tn+fp)
    if (tp+fp) == 0: ppv = np.nan
    else: ppv = tp/(tp+fp) # precision or positive predictive value (PPV)
    if (tn+fn) == 0: npv = np.nan
    else: npv = tn/(tn+fn) # negative predictive value (NPV)
    if (tp+tn+fp+fn) == 0: hitrate = np.nan
    else: hitrate = (tp+tn)/(tp+tn+fp+fn) # accuracy (ACC)
    performance = {'Accuracy': acc,
                   'AUC': auroc,
                   'WF1': f1,
                   'precision': precision,
                   'recall': recall,
                   'mcc': mcc,
                   'tp': tp,
                   'fp': fp,
                   'tn': tn,
                   'fn': fn,
                   'sensitivity': sensitivity,
                   'specificity': specificity,
                   'ppv': ppv,
                   'npv': npv,
                   'hitrate': hitrate,
                   'instances' : len(y_true)}
    return performance


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Loading PLIP Model...')
model = CLIPModel.from_pretrained("/root/model/plip")
processor = CLIPProcessor.from_pretrained("/root/model/plip")

data = pd.read_csv('/root/autodl-tmp/plip/reproducibility/ValData/DigestPath_test/DigestPath_test.csv')
images_neg = np.load('/root/autodl-tmp/plip/reproducibility/generate_validation_datasets/data_validation/DigestPath2019/Colonoscopy_tissue_segment_dataset/processed/cropsize=224_overlap=0.10_nonbgthreshold=0.50_downsamplelist=[2, 4, 8, 16, 32]/step_2_tumor2patch_ratio_threshold=0.50/final_negative_images.npy')
images_pos = np.load('/root/autodl-tmp/plip/reproducibility/generate_validation_datasets/data_validation/DigestPath2019/Colonoscopy_tissue_segment_dataset/processed/cropsize=224_overlap=0.10_nonbgthreshold=0.50_downsamplelist=[2, 4, 8, 16, 32]/step_2_tumor2patch_ratio_threshold=0.50/final_positive_images.npy')

rng = np.random.default_rng()
train_neg = rng.choice(images_neg, int(images_neg.shape[0]*0.7), replace=False)
rng = np.random.default_rng()
train_pos = rng.choice(images_pos, int(images_pos.shape[0]*0.7), replace=False)

train_dataset = np.concatenate((train_neg, train_pos))

rng = np.random.default_rng()
test_neg = rng.choice(images_neg, int(images_neg.shape[0]*0.3), replace=False)
rng = np.random.default_rng()
test_pos = rng.choice(images_pos, int(images_pos.shape[0]*0.3), replace=False)

test_dataset = np.concatenate((test_neg, test_pos))

test_y = [0]*int(images_neg.shape[0]*0.3)+[1]*int(images_pos.shape[0]*0.3)
train_y = [0]*int(images_neg.shape[0]*0.7)+[1]*int(images_pos.shape[0]*0.7)



train_embs = []
for img in tqdm(train_dataset):
    image = Image.fromarray(img)
    outputs = model(**processor(images=image, text=['image'], return_tensors="pt", padding=True))
    train_embs.append(outputs.image_embeds.detach().numpy().squeeze())
    
test_embs = []
for img in tqdm(test_dataset):
    image = Image.fromarray(img)
    outputs = model(**processor(images=image, text=['image'], return_tensors="pt", padding=True))
    test_embs.append(outputs.image_embeds.detach().numpy().squeeze())
  

for alpha in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
# for alpha in [1e-4, 6e-5, 5e-5, 4e-5, 1e-5]:
    classifier = SGDClassifier(random_state=1, loss="log_loss",
                               alpha=alpha, verbose=0,
                               penalty="l2", max_iter=10000, class_weight="balanced")
    le = LabelEncoder()
    train_y = le.fit_transform(train_y)
    test_y = le.transform(test_y)

    train_y = np.array(train_y)
    test_y = np.array(test_y)
    classifier.fit(train_embs, train_y)
    test_pred = classifier.predict(test_embs)
    # train_pred = classifier.predict(test_embs)
    test_metrics = eval_metrics(test_y, test_pred, average_method="macro")
    print(f'alpha: {alpha}')
    print(test_metrics)