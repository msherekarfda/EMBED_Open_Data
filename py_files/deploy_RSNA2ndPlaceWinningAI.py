'''
    Program converts all the dcm to jpg images first
'''
import os
import argparse
import datetime
import sys
import cv2
import glob
#import gdcm
import json
import shutil
import pydicom
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from distutils.util import strtobool

from sklearn.metrics import roc_curve, auc, roc_auc_score

import torch
import torch.nn.functional as F
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.types import DALIDataType
from pydicom.filebase import DicomBytesIO
from nvidia.dali.plugin.pytorch import feed_ndarray, to_torch_type

import dicomsdl
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from mmengine.runner import load_checkpoint
from mmengine.registry import MODELS
import mmengine
from mmcls.utils import register_all_modules
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from RSNA2ndPlaceDCM2JPG import *
from RSNA2ndPlaceInference import *
from RSNA2ndPlaceWinnerConstants import *

register_all_modules()
torch.backends.cudnn.benchmark = True
torch.jit.enable_onednn_fusion(True)


def deploy(args, exp_name):
    # # convert dcm to jpg
    print('Converting DCM ....')
    #dcm_to_jpg(args, args.dcm_root_dir_pth)
    
    # # inference
    print('Inferencing ....')
    preds_df, sub = inference(args, INPUT_CHK_PT, THRESHOLD, exp_name)
    
    # # clean-up
    if args.remove_processed_pngs:
        print('WARNING. DELETING ' + args.save_img_root_dir_pth)
        shutil.rmtree(args.save_img_root_dir_pth)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploying RSNA model on input data')
    parser.add_argument('--csv-pth', type=str, help='Path to CSV file')
    parser.add_argument('--dcm-root-dir-pth', type=str, help='Root directory of DICOM images')
    parser.add_argument('--save-img-root-dir-pth', type=str, help='Directory to save processed PNG images')
    parser.add_argument('--remove-processed-pngs', type=lambda x: bool(strtobool(x)), default = False, help = 'Choose if you want to save generated pngs (True or False)')
    parser.add_argument('--data', type=str, default = 'RSNA', choices = ['RSNA', 'EMBED'], help = 'Choose dataset')
    parser.add_argument('--out-dir', type=str, help='Output root directory')
    parser.add_argument('--weight_file', help='input weight file', required=True)
    parser.add_argument('--MC_dropout', default = False, type=lambda x: bool(strtobool(x)), help = 'Choose if you want apply MC dropout at test time')
    parser.add_argument('--train_MC_dropout', default = False, type=lambda x: bool(strtobool(x)), help = 'if the model applied MC dropout during training')
    parser.add_argument('--random_state', type=int, default=None)
    parser.add_argument('--dropout_rate', type=float, default=None)
    parser.add_argument('--test_time_augment', default = False, type=lambda x: bool(strtobool(x)), help = 'Choose if you want apply test time augmentation')
    args = parser.parse_args()
    print(args)
    # # save the args
    os.makedirs(args.out_dir, exist_ok=True)
    exp_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '__' +  os.path.basename(args.csv_pth)
    args_log_file = os.path.join(args.out_dir, exp_name + '__deploy_args.json')
    print(args_log_file)
    with open(args_log_file, 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    
    # # call deploy
    deploy(args, exp_name)
    print('.... COMPLETED')