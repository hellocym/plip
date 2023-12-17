# zeroshot validation reproduction for DigestPath
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import wandb
import random
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Loading PLIP Model...')
model = CLIPModel.from_pretrained("/root/model/plip")
processor = CLIPProcessor.from_pretrained("/root/model/plip")

data = pd.read_csv('/root/autodl-tmp/PanNuke/processed_threshold=10_0.3/PanNuke_all_binary.csv')

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="PLIP",
    name='PanNuke',
    # track hyperparameters and run metadata
    config={
    "architecture": "PLIP",
    "dataset": "PanNuke",
    "epochs": 3,
    }
)

from sklearn.metrics import f1_score

TP = TN = FP = FN = 0

gt = []
pr = []


for patch in range(623):
    d = data.sample(n=10)
    images = d['image'].tolist()
    captions = d['caption'].tolist()
    for img, cap in zip(images, captions):
        image = Image.open(img)
        text = [cap.replace('malignant', 'benign'), cap.replace('benign', 'malignant')]
        inputs = processor(text=text,
                           images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        pred = np.argmax(probs.detach().numpy())
        gt.append(0 if 'benign' in cap else 1)
        pr.append(pred)
        if pred == 0:
            # predicted as negative (benign)
            if 'benign' in cap:
                TN += 1
            elif 'malignant' in cap:
                FN += 1
        else:
            # predicted as positive (malignant)
            if 'benign' in cap:
                FP += 1
            elif 'malignant' in cap:
                TN += 1
                
    print(f'TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}.')
    Acc = (TP+TN)/(TP+FP+TN+FN)
    P = TP/(TP+FP) if (TP+FP) != 0 else np.nan
    R = TP/(TP+FN) if (TP+FN) != 0 else np.nan
    # F1 = 2*(P*R)/(P+R)
    # F1 = TP/(TP + 0.5*(FP+FN))
    F1W = f1_score(np.array(gt), np.array(pr),average='weighted')
    print(f'Accuracy: {Acc}')
    print(f'Precision: {P}')
    print(f'Recall: {R}')
    # print(f'F1 Score: {F1}')
    print(f'F1 Score Weighted: {F1W}')
    wandb.log({"acc": Acc, "F1W": F1W})
    # break
            

wandb.finish()