# imports
import os, sys
import argparse
import numpy as np

import torch
import torch.nn
from torch.utils.data import DataLoader

from dataset import SyntheticDataset
from model import EfficientPhys
from utils import _calculate_fft_hr, get_signal

# args
parser = argparse.ArgumentParser(description='Classification vs Regression')
parser.add_argument("--mode", type=str, help="Model mode", default="regression")
parser.add_argument("--lr", type=float, help="LR", default=1e-3)
parser.add_argument("--epochs", type=int, help="Num epochs", default=50)
args = parser.parse_args()

# create train and test dataloaders
train_dataset = SyntheticDataset("data/train.txt", args.mode)
test_dataset = SyntheticDataset("data/test.txt", args.mode)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

# create a model - classification or regression
model = EfficientPhys(3, 16, 32, 3, 0.5, 0.5, (2,2), 64, 150, 28, args.mode)
model = model.to('cuda') # move to GPU

# train and test
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

if args.mode == "classification":
    loss_fn = torch.nn.BCELoss()    # Binary Cross Entropy loss - logit (GT) and prediction score output

best_err = 1e8
for epoch in range(args.epochs):
    # set model to train
    model.train()

    tloss, vloss, fmae = 0.0, 0.0, 0.0
    for idx, data in enumerate(train_loader):
        vid, angle, freq = data
        # move to device
        vid, angle, freq = vid.cuda(), angle.cuda(), freq.cuda()

        N, D, C, H, W = vid.shape
        vid = vid.view(N * D, C, H, W)
        angle = angle.view(-1, 1)
        freq = freq.view(-1, 1)
        
        optimizer.zero_grad()
        # TODO: create a 10 frame sample from the input
        out = model(vid)

        # loss:
        loss = loss_fn(out, angle)
        loss.backward()

        optimizer.step()
        tloss += loss.item()

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            # TODO: Create another for loop to get kx10 frame windows for model inputs.
            # Using 150 frames as input to the model now.
            vid, angle, freq = data
            # move to device
            vid, angle, freq = vid.cuda(), angle.cuda(), freq.cuda()

            N, D, C, H, W = vid.shape
            vid = vid.view(N * D, C, H, W)
            angle = angle.view(-1, 1)
            freq = freq.view(-1, 1)
        
            out = model(vid)

            # loss 
            loss = loss_fn(out, angle)
            vloss += loss

            # frequency estimate
            out = out.cpu().detach().numpy().squeeze()
            if args.mode == "classification":
                out = get_signal(out)

            f_ = _calculate_fft_hr(out)
            #print(f_, freq.squeeze().item())
            fmae += np.abs(f_-freq.squeeze().item())
    
    if fmae <= best_err:
        best_err = fmae
        # save the model as checkpoint
        torch.save(model.state_dict(), "checkpoints/{}_model.pth".format(args.mode))

    print("Epoch:{epoch} : train loss: {tloss}, validation loss: {vloss}, freq MAE: {fmae}".format(
        epoch=epoch, tloss=tloss/len(train_loader), vloss=vloss/len(test_loader), fmae=fmae/len(test_loader)))
