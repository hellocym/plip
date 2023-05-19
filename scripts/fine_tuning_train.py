import sys
sys.path.append("../")
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import logging
import time
from embedders.factory import EmbedderFactory
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
opj = os.path.join
import numpy as np
from utils.results_handler import ResultsHandler

def torch_init(random_seed):
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def convert_dataset_labels(args, df):
    #TODO This is currently hard-coded. May need to refactorize.
    df = df[['image', 'label']] # this is hard-coded
    df['image'] = df['image'].replace('pathtweets_data_20230211', 'pathtweets_data_20230426')
    if args.dataset == 'Kather':
        label2digit = {'ADI':0, 'BACK':1, 'DEB':2, 'LYM':3, 'MUC':4, 'MUS':5, 'NORM':6, 'STR':7, 'TUM':8}
        df['label'] = df['label'].apply(lambda v: label2digit[v])
    elif args.dataset in ['DigestPath', 'PanNuke', 'WSSS4LUAD_binary']:
        df['label'] = df['label'].astype(int)
    else:
        raise Exception('No dataset available.')
    return df

def tune_clip(args, train, valid, logging):
    # re-initialize torch at every training.
    torch_init(args.random_seed)
    from fine_tuning.clip import CLIPTuner
    if args.model_name == 'clip':
        backbone = None
    elif args.model_name == "plip":
        backbone = args.backbone # re-defined in previous line.
    else:
        raise Exception('Error.')
    cpt = CLIPTuner(args=args,
                    logging=logging,
                    model_type=args.PC_CLIP_ARCH,
                    backbone=backbone,
                    num_classes=args.num_classes,
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                    px_size=args.pxsize,
                    comet_tracking=args.comet_tracking,
                    comet_tags=args.comet_tags
                    )

    performance = cpt.tuner(train, valid, save_directory=args.save_directory, batch_size=args.batch_size,
                            epochs=args.epochs, evaluation_steps=args.evaluation_steps, num_workers=args.num_workers)
    return performance
    
def config():
    load_dotenv("../config.env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="plip", type=str, choices=['plip', 'clip', 'mudipath'])
    parser.add_argument("--backbone", default='default', type=str)
    parser.add_argument("--dataset", default="Kather", type=str, choices=['Kather', 'PanNuke', 'WSSS4LUAD_binary', 'DigestPath'])

    ## Fine-tuning hparams
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--percentage_of_training_data", default=1.0, type=float,
                        help="""The ratio of the training data (range 0.0 - 1.0).
                                If value = 1, use all training data to fine-tune.
                                If value = 0.2, use 20%% of the training data to fine-tune.""")
    parser.add_argument("--valid_ratio", default=0.10, type=float,
                        help="""The ratio of the validation set that came from training data.
                                If sub-sampling was performed on the training data, the validation set is generated using the sub-sampled portion.""")
    # Deprecate learning-rate: set it in for loop.
    #parser.add_argument("--learning-rate", default=1e-5, type=float)
    parser.add_argument("--weight-decay", default=0.1, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--first_resize", default=512, type=int, help='This is image preprocessing transformation parameter.')
    parser.add_argument("--pxsize", default=224, type=int)
    parser.add_argument("--optimizer", default='AdamW', type=str)
    parser.add_argument("--evaluation-steps", default=0, type=int, help='set to 0 to disable the evalutation steps (only evaluate at the end of each epoch)')
    parser.add_argument("--save_directory", default='/oak/stanford/groups/jamesz/pathtweets/results/fine_tuning')
    parser.add_argument("--comet-tracking", default=False)
    parser.add_argument("--comet_tags", nargs="*")
    parser.add_argument("--random_seed", default=0, type=int)

    return parser.parse_args()


if __name__ == "__main__":

    args = config()
    np.random.seed(args.random_seed)
    data_folder = os.environ["PC_EVALUATION_DATA_ROOT_FOLDER"]
    args.PC_CLIP_ARCH = os.environ["PC_CLIP_ARCH"]
    if args.model_name == "plip" and args.backbone == "default":
        args.backbone = os.environ["PC_DEFAULT_BACKBONE"]
    
    print('Now working on:')
    print(f'    Dataset: {args.dataset}')
    print(f'    Model: {args.model_name}')
    print(f'    Backbone: {args.backbone}')

    ##########################################
    # Step 1. Prepare the dataset
    ##########################################

    train_dataset_name = args.dataset + "_train.csv"
    test_dataset_name = args.dataset + "_test.csv"

    train_dataset = pd.read_csv(os.path.join(data_folder, train_dataset_name))
    test_dataset = pd.read_csv(os.path.join(data_folder, test_dataset_name))

    train_dataset = convert_dataset_labels(args, train_dataset)
    test_dataset = convert_dataset_labels(args, test_dataset)
    args.num_classes = len(train_dataset['label'].unique())

    # Subsmple dataset
    if args.percentage_of_training_data < 1:
        print('Subsample dataset (few-shot)')
        print(f'Number of training data before sub-sampling: {len(train_dataset)}')
        train_dataset = train_dataset.sample(frac=args.percentage_of_training_data, random_state=args.random_seed)
        print(f'Number of training data after sub-sampling : {len(train_dataset)}')


    train, valid = train_test_split(train_dataset, test_size=args.valid_ratio,
                                            random_state=args.random_seed,
                                            shuffle=True)
                                            
    print(f'Number of training: {len(train)} / validation: {len(valid)} / testing: {len(test_dataset)}')
    
    TIMESTRING  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())
    if args.model_name == 'plip':
        savesubdir = f'PLIP_{TIMESTRING}_data={args.dataset}_btch={args.batch_size}_'+\
                        f'wd={args.weight_decay}_firstresize={args.first_resize}_pxsize={args.pxsize}_nepochs={args.epochs}_'+\
                        f'validratio={args.valid_ratio}_optimizer={args.optimizer}'
    else:
        savesubdir = f'{args.model_name}_{TIMESTRING}'
    args.save_directory = opj(args.save_directory, args.dataset, f'train_ratio={args.percentage_of_training_data}', savesubdir, f'random_seed={args.random_seed}')
    os.makedirs(args.save_directory, exist_ok=True)
    
    args_df = pd.DataFrame(vars(args),index=['Value']).T
    args_df.to_csv(opj(args.save_directory, 'arguments.csv'))
    
    print('------------------------------')
    print(args_df)
    print('------------------------------')

    
    logging.basicConfig(filename=opj(args.save_directory, '_training.log'),
                        format='%(asctime)s.%(msecs)03d *** %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        encoding='utf-8',
                        level=logging.INFO
                        )

    args.comet_tracking = None

    ##########################################
    # Step 2. Run Train-validation to find hyper-parameter
    ##########################################
    lr_search_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    all_performance = pd.DataFrame()


    for lr in lr_search_list:
        print(f'Current learning rate: {lr}')
        logging.info(f'Current learning rate: {lr}')
        args.learning_rate = lr
        if args.model_name in ["clip", "plip"]:
            performance = tune_clip(args, train, valid, logging)
            performance['learning_rate'] = lr
            all_performance = pd.concat([all_performance, performance], axis=0).reset_index(drop=True)
        elif args.model_name == 'EfficientNet':
            pass
            
    print(all_performance)
    # Evaluate at max epoch:
    perf_maxepoch = all_performance.loc[all_performance['epoch'] == args.epochs]
    best_lr = perf_maxepoch['learning_rate'][perf_maxepoch['f1_weighted'].idxmax()]

    print(f"Best learning rate: {best_lr}")
    logging.info(f"Best learning rate: {best_lr}")

    ##########################################
    # Step 3. Use the best hyperparameter and retrain the model
    #         by combining training and validation split.
    ##########################################
    args.learning_rate = best_lr

    if args.model_name in ["clip", "plip"]:
        performance_test = tune_clip(args, train_dataset, test_dataset, logging)
    elif args.model_name == 'EfficientNet':
        pass

    print(performance_test)