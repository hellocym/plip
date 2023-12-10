from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import wandb
import random
from sklearn.metrics import f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("model/plip")
processor = CLIPProcessor.from_pretrained("model/plip")

data = pd.read_csv('/root/autodl-tmp/plip/reproducibility/ValData/DigestPath_test/DigestPath_test.csv'
images_neg = np.load('/root/autodl-tmp/plip/reproducibility/generate_validation_datasets/data_validation/DigestPath2019/Colonoscopy_tissue_segment_dataset/processed/cropsize=224_overlap=0.10_nonbgthreshold=0.50_downsamplelist=[2, 4, 8, 16, 32]/step_2_tumor2patch_ratio_threshold=0.50/final_negative_images.npy')
images_pos = np.load('/root/autodl-tmp/plip/reproducibility/generate_validation_datasets/data_validation/DigestPath2019/Colonoscopy_tissue_segment_dataset/processed/cropsize=224_overlap=0.10_nonbgthreshold=0.50_downsamplelist=[2, 4, 8, 16, 32]/step_2_tumor2patch_ratio_threshold=0.50/final_positive_images.npy')

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="PLIP",
    
    # track hyperparameters and run metadata
    config={
    "architecture": "PLIP",
    "dataset": "DigestPath",
    "epochs": 940*2,
    }
)

TP = TN = FP = FN = 0

gt = []
pr = []


for patch in range(940):
    rng = np.random.default_rng()
    neg_sampled = rng.choice(images_neg, 10)
    for img in neg_sampled:
        image = Image.fromarray(img)
        text = ["An H&E image patch of benign tissue.", "An H&E image patch of malignant tissue."]
        inputs = processor(text=text,
                           images=image, return_tensors="pt", padding=True)
        inputs = inputs
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        pred = np.argmax(probs.detach().numpy())
        gt.append(0)
        pr.append(pred)
        if pred == 0:
            # predicted as negative (benign)
            TN += 1
        else:
            # predicted as positive (malignant)
            FP += 1

    rng = np.random.default_rng()
    pos_sampled = rng.choice(images_pos, 10)        
    for img in pos_sampled:
        image = Image.fromarray(img)
        text = ["An H&E image patch of benign tissue.", "An H&E image patch of malignant tissue."]
        inputs = processor(text=text,
                           images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        pred = np.argmax(probs.detach().numpy())
        gt.append(1)
        pr.append(pred)
        if pred == 0:
            # predicted as negative (benign)
            FN += 1
        else:
            # predicted as positive (malignant)
            TP += 1
            
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

wandb.finish()