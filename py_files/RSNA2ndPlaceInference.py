import cv2
import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from mmengine.runner import load_checkpoint
from mmengine.registry import MODELS
import numpy as np
import pandas as pd
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

from test_time_augmentation import *

from PIL import Image
import torchvision.transforms as T

class BreastDataset(Dataset):
    """
    Image torch Dataset.
    """
    def __init__(
        self,
        df,
        transforms=None,
    ):
        """
        Constructor

        Args:
            paths (list): Path to images.
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
        """
        self.paths = df['path'].values
        self.labels = df['cancer'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Item accessor

        Args:
            idx (int): Index.

        Returns:
            np array [H x W x C]: Image.
            torch tensor [1]: Label.
            torch tensor [1]: Sample weight.
        """
        image = cv2.imread(self.paths[idx])

        image = self.transforms(image=image)["image"]
 
        #transfo = T.ToPILImage()
        #img = transfo(image)
        #img.save(os.path.join('/scratch/yuhang.zhang/EMBED_OUT/2023_MLDrift/uncertainty/example_augment_image_erase/',os.path.basename(self.paths[idx])))
        label = self.labels[idx]

        return image, label


def predict(model, dataset, batch_size=64, device="cuda"):
    """
    Torch predict function.

    Args:
        model (torch model): Model to predict with.
        dataset (CustomDataset): Dataset to predict on.
        loss_config (dict): Loss config, used for activation functions.
        batch_size (int, optional): Batch size. Defaults to 64.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(dataset) x num_classes]: Predictions.
    """
    model.eval()
    preds = torch.empty(0, dtype=torch.float, device='cuda')

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(),pin_memory=True,
        prefetch_factor=4,persistent_workers=True
    )
    norm_fn = torchvision.transforms.Normalize(mean=[77.52425988, 77.52425988, 77.52425988],
    std=[51.8555656, 51.8555656, 51.8555656])
    with torch.no_grad():
        for batch in loader:
            
            x = batch.to(device)
            batch_c = x.shape[0]
            # TTA hflip vflip
            x_fliph = torchvision.transforms.functional.hflip(x.clone())
            x = torch.cat([x,x_fliph], dim=0)
            x = norm_fn(x.float())
            # Forward
            pred = model(x)
            pred = pred.softmax(-1)[:,1]
            
            pred = (pred[:batch_c]+pred[batch_c:batch_c*2])/2
            preds = torch.cat([preds, pred], dim=0)

    return preds.cpu().numpy()
        
def append_dropout(model, rate):
    """ Function to add dropout layer after each convolutional layer"""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module, rate)
        if isinstance(module, torch.nn.Conv2d):
            new = torch.nn.Sequential(module, torch.nn.Dropout2d(p=rate, inplace=True))
            setattr(model, name, new)    
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            
def enable_droppath(model):
    """ Function to enable the droppath during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('DropPath'):
            m.train()   

def inference(args, INPUT_CHK_PT, THRESHOLD, exp_name):
    if args.random_state is not None:
        torch.manual_seed(args.random_state)
    # # inference
    model_dict = dict(
        type='ImageClassifier',
        backbone=dict(
            type='ConvNeXt',
            arch='small',
            out_indices=(3, ),
            drop_path_rate=0.4,
            gap_before_final_norm=True,
            init_cfg=[
                dict(
                    type='TruncNormal',
                    layer=['Conv2d', 'Linear'],
                    std=0.02,
                    bias=0.0),
                dict(type='Constant', layer=['LayerNorm'], val=1.0, bias=0.0)
            ]),
            head=dict(
                type='LinearClsHead',
                num_classes=2,
                in_channels=768,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
    #/projects01/didsr-aiml/common_data/EMBED/images/cohort_1/10000879/1.2.842.113970.3.62.1.56868341.20180426.1095160/1.2.844.113684.2750825168.1524118972.4854.25053/1.2.826.0.1.3680043.8.498.10392068038916878965464813474172245832.dcm
    df = pd.read_csv(args.csv_pth)
    # print(df.head())
    if args.data == 'RSNA':
        df['path'] = args.save_img_root_dir_pth + df["patient_id"].astype(str) + "_" + df["image_id"].astype(str) + ".png"
    elif args.data == 'EMBED':
        df['path'] = df["image_id"]
        df['path'] = df['path'].str.split(os.sep).str[1:]
        df['path'] = df['path'].str.join('_')
        df['path'] = args.save_img_root_dir_pth + df['path'].astype(str) + ".png"
    elif args.data == 'CBIS':
        df['path'] = df["image file path"]
        df['path'] = df['path'].str.split(os.sep).str[1:]
        df['path'] = df['path'].str.join('_')
        df['path'] = args.save_img_root_dir_pth + df['path'].astype(str) + ".png"
    else:
        print('!ERROR. UNKNOWN DATA TYPE. NOTHING TO DO. EXITING')
        return
    
    model = MODELS.build(model_dict).to('cuda').eval()
    if args.train_MC_dropout:
        append_dropout(model, args.dropout_rate)
    load_checkpoint(model, args.weight_file)
    # # apply MC dropout if required
    if args.MC_dropout:                                    
        enable_dropout(model)
        print(model)
        
    #enable_droppath(model)

    dataset = BreastDataset(
        df,
        transforms=get_transforms(augment=args.test_time_augment),
    )

    preds = torch.empty(0, dtype=torch.float, device='cuda')

    val_loader = DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0)
    norm_fn = torchvision.transforms.Normalize(mean=[77.52425988, 77.52425988, 77.52425988],
                                                std=[51.8555656, 51.8555656, 51.8555656])
    model.eval()                                            
    type_all = []
    logits_all = []
    scores_all = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            # # compute output
            images = images.cuda()
            ## >>
            #images_fliph = torch.flip(images, dims=[2,3])
            #images = torch.cat([images,images_fliph], dim=0)
            images = norm_fn(images.float())
            ## <<
            # output = model(images.float())
            output = model(images)
            output = output[:,1]
            # #
            target_image_pred_logits = torch.flatten(output)
            target_image_pred_probs = torch.sigmoid(target_image_pred_logits)
            # # accumulate the scores
            labl_list = list(target.cpu().numpy())
            type_all += labl_list
            #pid_all += pid
            #fnames_all += fname
            logit = list(target_image_pred_logits.cpu().numpy())
            logits_all += logit
            scr = list(target_image_pred_probs.cpu().numpy())
            scores_all += scr
            
    fpr, tpr, _ = roc_curve(np.array(type_all), np.array(scores_all), pos_label=1)
    auc_val = auc(fpr, tpr)
    print('AUROC = ' + str(auc_val))
    # # summarize and save results
    df["scores"] = scores_all
    df["label"] = type_all
    
    submit_df = df[['patient_id', 'study_date_anon_x', 'laterality', 'scores', 'label']]
    
    # # Simple avg for per-breast for an exam prediction
    submit_df = submit_df.groupby(['patient_id', 'study_date_anon_x', 'laterality']).mean()
    # # AUC from ROC
    #fpr, tpr, _ = roc_curve(np.array(submit_df['label']), np.array(submit_df['scores']), pos_label=1)
   # auc_val = auc(fpr, tpr)
    #print('AUROC = ' + str(auc_val))
    
    # # save
    df.to_csv(os.path.join(args.out_dir, 'input_list_file_with_output_scores.csv'), index=False)
    submit_df.to_csv(os.path.join(args.out_dir, 'by_patient_scores.csv'), index=False)
    
    # # ROC curve plotting
    # fpr, tpr, thresholds = roc_curve(df['cancer'], df["preds"])
    # auc_score = roc_auc_score(df['cancer'], df['preds'])
    # print(f"AUC: {auc_score:.3f}")
    
    # plt.figure(figsize=(6, 6), dpi=600)
    # plt.plot(fpr, tpr, color = 'darkorange', lw=3, label = f'AUC: {auc_score:.3f}')
    # plt.plot([0,1], [0,1], color='navy', lw = 2, linestyle = '--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate', weight = 'bold')
    # plt.ylabel('True Positive Rate', weight = 'bold')
    # plt.title('Receiver Operating Characteristic', weight = 'bold')
    # plt.legend(loc = "lower right")
    # plt.savefig(os.path.join(args.out_dir, exp_name + '__ROC.png'), dpi=600, bbox_inches='tight')
    
    return df, submit_df