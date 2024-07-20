import cv2
import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from torch.utils.tensorboard import SummaryWriter
# from torchsummaryX import summary
from mmengine.runner import load_checkpoint, save_checkpoint
from mmengine.registry import MODELS
import numpy as np
import pandas as pd
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from mmengine.dist import all_reduce as allreduce

from test_time_augmentation import *
from finite_sample import *

import datetime

import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
master_iter = 0

norm_fn = torchvision.transforms.Normalize(mean=[77.52425988, 77.52425988, 77.52425988],
                                           std=[51.8555656, 51.8555656, 51.8555656])


class SoftmaxEQLLoss(_Loss):
    def __init__(self, num_classes, indicator='pos', loss_weight=1.0, tau=1.0, eps=1e-4):
        super(SoftmaxEQLLoss, self).__init__()
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.tau = tau
        self.eps = eps

        assert indicator in ['pos', 'neg', 'pos_and_neg'], 'Wrong indicator type!'
        self.indicator = indicator

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(num_classes))
        self.register_buffer('neg_grad', torch.zeros(num_classes))
        self.register_buffer('pos_neg', torch.ones(num_classes))

    def forward(self, input, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if self.indicator == 'pos':
            indicator = self.pos_grad.detach()
        elif self.indicator == 'neg':
            indicator = self.neg_grad.detach()
        elif self.indicator == 'pos_and_neg':
            indicator = self.pos_neg.detach() + self.neg_grad.detach()
        else:
            raise NotImplementedError

        one_hot = F.one_hot(label.long(), self.num_classes)
        self.targets = one_hot.detach()

        matrix = indicator[None, :].clamp(min=self.eps) / indicator[:, None].clamp(min=self.eps)
        factor = matrix[label.detach().cpu().long(), :].pow(self.tau)

        factor = factor.to(device='cuda')

        cls_score = input + (factor.log() * (1 - one_hot.detach()))
        loss = F.cross_entropy(cls_score, label.long())

        prob = torch.sigmoid(cls_score)
        #prob = torch.softmax(cls_score)
        grad = self.targets * (prob - 1) + (1 - self.targets) * prob
        self.collect_grad(grad)

        return loss * self.loss_weight

    def collect_grad(self, grad):
        grad = torch.abs(grad)
        pos_grad = torch.sum(grad * self.targets, dim=0)
        neg_grad = torch.sum(grad * (1 - self.targets), dim=0)

        allreduce(pos_grad)
        allreduce(neg_grad)

        self.pos_grad += pos_grad.detach().cpu()
        self.neg_grad += neg_grad.detach().cpu()
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)


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

        label = self.labels[idx]

        return image, label


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def predict(model, dataset, batch_size=4, device="cuda"):
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
        dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True,
        prefetch_factor=4, persistent_workers=True
    )
    norm_fn = torchvision.transforms.Normalize(mean=[77.52425988, 77.52425988, 77.52425988],
                                               std=[51.8555656, 51.8555656, 51.8555656])
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            batch_c = x.shape[0]
            # TTA hflip vflip
            x_fliph = torchvision.transforms.functional.hflip(x.clone())
            x = torch.cat([x, x_fliph], dim=0)
            x = norm_fn(x.float())
            # Forward
            pred = model(x)
            pred = pred.softmax(-1)[:, 1]

            pred = (pred[:batch_c] + pred[batch_c:batch_c * 2]) / 2
            preds = torch.cat([preds, pred], dim=0)

    return preds.cpu().numpy()


def load_data_from_csv(args, csv_pth, training_flag=False):
    df = pd.read_csv(csv_pth)
    if training_flag and (args.finite_sample_rate is not None):
        df = data_sampling(df, args.finite_sample_rate)
        df.to_csv(os.path.join(args.out_dir, f'training_list_sample_rate_{args.finite_sample_rate}.csv'), index=False)
    #df = data_filtering(df, 'tissueden', [1,2])
    #df.to_csv(os.path.join(args.out_dir, f'training_list.csv'), index=False)
    print(df.head())
    if args.data == 'RSNA':
        df['path'] = args.save_img_root_dir_pth + df["patient_id"].astype(str) + "_" + df["image_id"].astype(
            str) + ".png"
    elif args.data == 'EMBED':
        df['path'] = df["image_id"]
        df['path'] = df['path'].str.split(os.sep).str[1:]
        df['path'] = df['path'].str.join('_')
        df['path'] = args.save_img_root_dir_pth + df['path'].astype(str) + ".png"
        #print('path', df['path'][0])
    elif args.data == 'CBIS':
        df['path'] = df["image file path"]
        df['path'] = df['path'].str.split(os.sep).str[1:]
        df['path'] = df['path'].str.join('_')
        df['path'] = args.save_img_root_dir_pth + df['path'].astype(str) + ".png"
    else:
        raise RuntimeError('!ERROR. UNKNOWN DATA TYPE. NOTHING TO DO. EXITING')
    if training_flag and args.training_augment:
        dataset = BreastDataset(df, transforms=get_transforms(augment=True))
        print("training augmented")
    else:
        dataset = BreastDataset(df, transforms=get_transforms(augment=False))

    return dataset


def apply_custom_transfer_learning(model, num_to_freeze):
    model_layers = [name for name, para in model.named_parameters()]

    for ii in range(42):
        if ii == 0:
            layers = [','.join(model_layers[:3])]
        elif ii <= 3 and ii > 0:
            layers += [','.join(model_layers[(ii * 4):(ii * 4 + 4)])]
        elif ii <= 39 and ii > 3:
            layers += [','.join(model_layers[(16 + (ii - 4) * 9):(25 + (ii - 4) * 9)])]
        else:
            layers += [','.join(model_layers[(340 + (ii - 40) * 2):(342 + (ii - 40) * 2)])]
    fine_tune_layers = ','.join(layers[num_to_freeze - len(layers):]).split(',')
    for name, param in model.named_parameters():
        #print(name)        
        if name not in fine_tune_layers:
            print(name)
            param.requires_grad = False

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    print([len(parameters), len(fine_tune_layers)])
    assert len(parameters) == len(fine_tune_layers)

    return model


def append_dropout(model, rate):
    """ Function to add dropout layer after each convolutional layer"""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module, rate)
        if isinstance(module, torch.nn.Conv2d):
            new = torch.nn.Sequential(module, torch.nn.Dropout2d(p=rate, inplace=True))
            setattr(model, name, new)


def train_model(args, INPUT_CHK_PT, THRESHOLD, exp_name):
    # # start training
    print("Model initialization ...", flush=True)
    # set random state, if specified
    if args.random_state is not None:
        torch.manual_seed(args.random_state)
    # # loading model
    model_dict = dict(
        type='ImageClassifier',
        backbone=dict(
            type='ConvNeXt',
            arch='small',
            out_indices=(3,),
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

    # model = MODELS.build(model_dict).to('cuda').eval()
    model = MODELS.build(model_dict).to('cuda')
    # model.cuda(4)
    print("Loading checkpoint ...", flush=True)
    load_checkpoint(model, INPUT_CHK_PT)

    # # apply dropout if required
    if args.dropout_rate is not None:
        append_dropout(model, args.dropout_rate)
        print(model)

        # # freeze first number of layers if specified
    if args.fine_tuning == 'partial':
        model = apply_custom_transfer_learning(model, args.upto_freeze)

    #x=torch.rand(1,3,1536,1536)
    #print(summary(model, x))
    # # load training and validation dataset
    print("Loading data ...", flush=True)
    train_dataset = load_data_from_csv(args, args.train_csv_pth, training_flag=True)
    valid_dataset = load_data_from_csv(args, args.valid_csv_pth, training_flag=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    # norm_fn = torchvision.transforms.Normalize(mean=[77.52425988, 77.52425988, 77.52425988],
    #                                             std=[51.8555656, 51.8555656, 51.8555656])
    ## >>
    #criterion = nn.BCEWithLogitsLoss()
    criterion = SoftmaxEQLLoss(num_classes=2)
    # # select the optimizer
    if args.optimizer == 'adam':
        # optimizer = torch.optim.Adam(model.parameters(), args.start_learning_rate, weight_decay=args.step_decay)
        optimizer = torch.optim.Adam(model.parameters(), args.start_learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.start_learning_rate, momentum=args.SGDmomentum)
    else:
        print('ERROR. UNKNOWN optimizer.')
        return
    # # learning rate scheduler
    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_every_N_epoch,
                                                      gamma=args.decay_multiplier)

    ## <<
    print("Start epoch ...", flush=True)
    print(model)
    writer = []
    for epoch in range(args.num_epochs):
        # # train for one epoch
        start_time = datetime.datetime.now().replace(microsecond=0)
        avg_loss = run_train(train_loader, model, criterion, optimizer, my_lr_scheduler, writer)
        my_lr_scheduler.step()
        my_lr = my_lr_scheduler.get_last_lr()
        # # save
        if epoch % args.save_every_N_epochs == 0 or epoch == args.num_epochs - 1:
            # # TODO evaluate on validation set
            auc_val = run_validate(valid_loader, model, args, writer)
            end_time = datetime.datetime.now().replace(microsecond=0)
            time_diff = end_time - start_time
            print(f"> {epoch} {round(avg_loss, 4)} {round(auc_val, 4)} {my_lr} {time_diff}", flush=True)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'ConvNeXt',
                'state_dict': model.state_dict(),
                'auc': auc_val,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.out_dir, 'checkpoint__' + str(epoch) + '.pth.tar'))
    # # save the last epoch model for deployment    
    last_model_path = os.path.join(args.out_dir, 'last_epoch_model.pth')
    save_checkpoint({
        'arch': 'ConvNeXt',
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, last_model_path)


def run_train(train_loader, model, criterion, optimizer, my_lr_scheduler, writer):
    """ Function that runs the training
    """
    global master_iter

    # switch to train mode
    model.train()
    avg_loss = 0
    for i, (images, target) in enumerate(train_loader):
        # # measure data loading time
        master_iter += 1
        images = images.cuda()
        # # image flip  amd concat are used in inference code
        # # decide not to do in training process
        #images_fliph = torch.flip(images, dims=[2,3])
        #images = torch.cat([images,images_fliph], dim=0)
        images = norm_fn(images.float())
        target = target.cuda()
        optimizer.zero_grad()
        output = model(images)
        #output = output[:,1]

        # # compute loss
        # loss = criterion(torch.sigmoid(torch.flatten(output)), target.float())
        loss = criterion(output, target.float())
        # writer.add_scalar("Loss/train", loss.item(), master_iter)
        avg_loss += loss.item()
        # # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        #my_lr_scheduler.step()
        # #
        # writer.add_scalar("LR/train", my_lr_scheduler.get_last_lr()[0], master_iter)
    return avg_loss / len(train_loader)


def run_validate(val_loader, model, args, writer):
    """ Function that deploys on the input data loader, calculates sample based AUC and saves the scores in a tsv file.
    """
    global master_iter

    # # switch to evaluate mode
    model.eval()
    # #
    #pid_all = []
    #fnames_all = []
    type_all = []
    logits_all = []
    scores_all = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            # # compute output
            try:
                images = images.cuda()
                ## >>
                #images_fliph = torch.flip(images, dims=[2,3])
                #images = torch.cat([images,images_fliph], dim=0)
                images = norm_fn(images.float())
                ## <<
                # output = model(images.float())
                output = model(images)
                output = output[:, 1]
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
                pass
            except Exception as e:
                print("error processing image batch {i}: {e}")
                continue

    # # save the scores, labels in a tsv file
    result_df1 = pd.DataFrame(list(zip(type_all, logits_all, scores_all)), columns=['label', 'logits', 'score'])
    if args.bsave_valid_results_at_epochs:
        results_path1 = os.path.join(args.out_dir, 'results__' + str(master_iter + 1) + '.tsv')
        result_df1.to_csv(results_path1, sep='\t', index=False)
    results_path2 = os.path.join(args.out_dir, 'results__last.tsv')
    result_df1.to_csv(results_path2, sep='\t', index=False)
    # # calc AUC from ROC
    fpr, tpr, _ = roc_curve(np.array(type_all), np.array(scores_all), pos_label=1)
    auc_val = auc(fpr, tpr)
    with open(os.path.join(args.out_dir, 'log.log'), 'a') as fp:
        fp.write("{:d}\t{:1.5f}\n".format(master_iter, auc_val))
    # writer.add_scalar("AUC/test", auc_val, master_iter)
    return auc_val


def train_model_loss_attenuation(args, INPUT_CHK_PT, THRESHOLD, exp_name):
    '''
    training with learned loss attenuation for aleatoric uncertainty estimation
    '''
    # # start training
    print("Model initialization ...", flush=True)
    # set random state, if specified
    if args.random_state is not None:
        torch.manual_seed(args.random_state)
    # # loading model
    model_dict = dict(
        type='ImageClassifier',
        backbone=dict(
            type='ConvNeXt',
            arch='small',
            out_indices=(3,),
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
            num_classes=1,
            in_channels=768,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))

    # model = MODELS.build(model_dict).to('cuda').eval()
    model = MODELS.build(model_dict).to('cuda')
    # model.cuda(4)
    print("Loading checkpoint ...", flush=True)
    load_checkpoint(model, INPUT_CHK_PT)

    # # add variance as model output
    for key, module in model.named_children():
        if key == 'head':
            module.fc = nn.Linear(768, 2)

    # # apply dropout if required
    if args.dropout_rate is not None:
        append_dropout(model, args.dropout_rate)
        print(model)

        # # freeze first number of layers if specified
    if args.fine_tuning == 'partial':
        model = apply_custom_transfer_learning(model, args.upto_freeze)

    model.to('cuda')
    # # load training and validation dataset
    print("Loading data ...", flush=True)
    train_dataset = load_data_from_csv(args, args.train_csv_pth, training_flag=True)
    valid_dataset = load_data_from_csv(args, args.valid_csv_pth, training_flag=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    # norm_fn = torchvision.transforms.Normalize(mean=[77.52425988, 77.52425988, 77.52425988],
    #                                             std=[51.8555656, 51.8555656, 51.8555656])
    ## >>
    criterion = nn.BCELoss(reduction='none')
    # # select the optimizer
    if args.optimizer == 'adam':
        # optimizer = torch.optim.Adam(model.parameters(), args.start_learning_rate, weight_decay=args.step_decay)
        optimizer = torch.optim.Adam(model.parameters(), args.start_learning_rate, betas=(0.9, 0.999),
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.start_learning_rate, momentum=args.SGDmomentum)
    else:
        print('ERROR. UNKNOWN optimizer.')
        return
    # # learning rate scheduler
    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_every_N_epoch,
                                                      gamma=args.decay_multiplier)

    ## <<
    print("Start epoch ...", flush=True)
    print(model)
    writer = []
    for epoch in range(args.num_epochs):
        # # train for one epoch
        start_time = datetime.datetime.now().replace(microsecond=0)
        avg_loss = run_train_loss_attenuation(train_loader, model, criterion, optimizer, my_lr_scheduler, writer,
                                              args.t_number)
        my_lr_scheduler.step()
        my_lr = my_lr_scheduler.get_last_lr()
        # # save
        if epoch % args.save_every_N_epochs == 0 or epoch == args.num_epochs - 1:
            auc_val = run_validate_loss_attenuation(valid_loader, model, args, writer)
            end_time = datetime.datetime.now().replace(microsecond=0)
            time_diff = end_time - start_time
            print(f"> {epoch} {round(avg_loss, 4)} {round(auc_val, 4)} {my_lr} {time_diff}", flush=True)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'ConvNeXt',
                'state_dict': model.state_dict(),
                'auc': auc_val,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.out_dir, 'checkpoint__' + str(epoch) + '.pth.tar'))
    # # save the last epoch model for deployment    
    last_model_path = os.path.join(args.out_dir, 'last_epoch_model.pth')
    save_checkpoint({
        'arch': 'ConvNeXt',
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, last_model_path)


def run_train_loss_attenuation(train_loader, model, criterion, optimizer, my_lr_scheduler, writer, t_num):
    """ Function that runs the training with learned loss attenuation
    """
    global master_iter

    # switch to train mode
    model.train()
    avg_loss = 0
    for i, (images, target) in enumerate(train_loader):
        # # measure data loading time
        master_iter += 1
        images = images.cuda()
        # # image flip  amd concat are used in inference code
        # # decide not to do in training process
        #images_fliph = torch.flip(images, dims=[2,3])
        #images = torch.cat([images,images_fliph], dim=0)
        images = norm_fn(images.float())
        target = target.cuda()
        optimizer.zero_grad()
        output = model(images)
        mu, sigma = output.split(1, 1)

        # # monte carlo sampling for learned loss attenuation
        loss_total = torch.zeros(t_num, target.size(0))
        for t in range(t_num):
            # assume that each logit value is drawn from Gaussian distribution, therefore the whole logit vector is drawn from multi-dimensional Gaussian distribution
            epsilon = torch.randn(sigma.size()).cuda()
            logit = mu + torch.mul(sigma.pow(2), epsilon)
            # # compute loss for each monte carlo sample
            loss_total[t] = criterion(torch.sigmoid(torch.flatten(logit)), target.float())
        # # compute average loss
        sample_loss = torch.mean(loss_total, 0)
        loss = torch.mean(sample_loss)
        avg_loss += loss.item()
        # # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
    return avg_loss / len(train_loader)


def run_validate_loss_attenuation(val_loader, model, args, writer):
    """ 
    """
    global master_iter

    # # switch to evaluate mode
    model.eval()
    # #
    #pid_all = []
    #fnames_all = []
    type_all = []
    logits_all = []
    scores_all = []
    sigma_all = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            # # compute output
            images = images.cuda()
            ## >>
            #images_fliph = torch.flip(images, dims=[2,3])
            #images = torch.cat([images,images_fliph], dim=0)
            images = norm_fn(images.float())
            ## <<
            output = model(images)
            test_mu, test_sigma = output.split(1, 1)

            output = test_mu
            sigma_out = torch.flatten(test_sigma)
            sigma = list(sigma_out.cpu().numpy())
            sigma_all += sigma
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

    # # save the scores, labels in a tsv file
    result_df1 = pd.DataFrame(list(zip(type_all, logits_all, scores_all, sigma_all)),
                              columns=['label', 'logits', 'score', 'uncertainty'])
    if args.bsave_valid_results_at_epochs:
        results_path1 = os.path.join(args.out_dir, 'results__' + str(master_iter + 1) + '.tsv')
        result_df1.to_csv(results_path1, sep='\t', index=False)
    results_path2 = os.path.join(args.out_dir, 'results__last.tsv')
    result_df1.to_csv(results_path2, sep='\t', index=False)
    # # calc AUC from ROC
    fpr, tpr, _ = roc_curve(np.array(type_all), np.array(scores_all), pos_label=1)
    auc_val = auc(fpr, tpr)
    with open(os.path.join(args.out_dir, 'log.log'), 'a') as fp:
        fp.write("{:d}\t{:1.5f}\n".format(master_iter, auc_val))
    # writer.add_scalar("AUC/test", auc_val, master_iter)

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_val:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(args.out_dir, 'roc_curve.png'))
    plt.show()

    # Plot score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(result_df1['score'], bins=30, kde=True)
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(args.out_dir, 'score_distribution.png'))
    plt.show()

    # Plot uncertainty distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(result_df1['uncertainty'], bins=30, kde=True)
    plt.title('Uncertainty Distribution')
    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(args.out_dir, 'uncertainty_distribution.png'))
    plt.show()

    return auc_val
