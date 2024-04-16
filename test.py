#imports
import os, sys
import argparse
from tqdm import tqdm
import math

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader

from dataset import SyntheticDataset
from model import EfficientPhys
from utils import _calculate_fft_hr, get_signal, _detrend, read_video, read_wave

# create an arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--mode', help="model mode", default="regression")
parser.add_argument('--input_path', help="input video path or leave blank for testing on test data", default="")
parser.add_argument('--weight_path', help="checkpoint path", default="checkpoints/regression_model.pth")
parser.add_argument('--detrend', help="detrend the signal", default=False)
args = parser.parse_args()

# load the checkpoint from model
model = EfficientPhys(3, 16, 32, 3, 0.5, 0.5, (2,2), 64, 150, 28, args.mode)
checkpoint = torch.load(args.weight_path)
model.load_state_dict(checkpoint) #['model_state_dict'])
model = model.cuda()
model.eval()

if args.input_path:
    # read video and convert to tensor for inference
    frames = read_video(args.input_path)
    frames = torch.from_numpy(np.float32(frames)) # TODO: Check how to convert to tensor.
    frames = torch.permute(frames, (0,3,1,2)) # (T, H, W, C) -> (T, C, H, W)

    # read GT freq and angle
    label_path = args.input_path.replace("mp4", "h5")
    angle, freq = read_wave(label_path)

    # inference
    frames = frames.cuda().contiguous()     # add contiguous to make the input data contiguous
    with torch.no_grad():
        out_angle = model(frames).cpu().detach().numpy().squeeze()

    if args.mode == "classification":
        out_angle = get_signal(out_angle)
    out_freq = _calculate_fft_hr(out_angle)

    # plot the out_angle, angle, write freq, and out_freq on the plot
    plt.plot(out_angle, label="Out")
    plt.plot(angle, label="GT")
    # write freq and out_freq on the plot at the top right
    plt.text(0.8, 0.9, "GT freq: " + str(freq), transform=plt.gca().transAxes)
    plt.text(0.8, 0.8, "Out freq: " + str(out_freq), transform=plt.gca().transAxes)

    # save the plot as inference_plot.png
    plt.savefig("inference_plot.png")
else:
    test_dataset = SyntheticDataset("data/test.txt", args.mode)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    preds, labels = [], [] 

    # plotting errors based on frequency range.
    freq_dict = {0:(0,0), 1:(0,0), 2:(0,0), 3:(0,0), 4:(0,0)}

    with torch.no_grad():
        for idx, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            # TODO: Create another for loop to get kx10 frame windows for model inputs.
            # Using 150 frames as input to the model now.
            vid, angle, freq, fname = data
            # move to device
            vid, angle, freq = vid.cuda(), angle.cuda(), freq.cuda()

            N, D, C, H, W = vid.shape
            vid = vid.view(N * D, C, H, W)
            angle = angle.view(-1, 1)
            freq = freq.view(-1, 1)
        
            out = model(vid)

            # frequency estimate
            out = out.cpu().detach().numpy().squeeze()
            if args.mode == "classification":
                out = get_signal(out)
                # check if detrending helps
                if args.detrend:
                    out = _detrend(out, 100)

            f_ = _calculate_fft_hr(out)
            gt = freq.squeeze().item()
            
            # update freq_dict
            cur, cnt = freq_dict[math.floor(gt)] 
            freq_dict[math.floor(gt)] = (cur+np.abs(f_-gt), cnt+1)

            if gt > 40.0:
                # plot the out_angle, angle, write freq, and out_freq on the plot
                plt.plot(out, label="Out")
                plt.plot(angle.cpu().detach().numpy().squeeze(), label="GT")

                # write freq and out_freq on the plot at the top right
                plt.text(0.8, 0.9, "GT freq: " + str(gt), transform=plt.gca().transAxes)
                plt.text(0.8, 0.8, "Out freq: " + str(f_), transform=plt.gca().transAxes)
                plt.legend(loc="upper left")
                
                # save the plot as inference_plot.png
                plt.savefig("plots/"+fname[0].split("/")[-1].replace(".mp4", ".png"))
                plt.clf()

            #print(f_, freq.squeeze().item())
            #fmae += np.abs(f_-gt)
            preds.append(f_)
            labels.append(gt)

    print("MAE in different frequency bands")    
    print([cur/cnt for cur, cnt in freq_dict.values()])

    print("MAE: ", np.mean(np.abs(np.array(preds)-np.array(labels))))
    print("RMSE: ", np.sqrt(np.mean(np.square(np.array(preds)-np.array(labels)))))
    print("Pearson Correlation: ", np.corrcoef(preds, labels)[0][1])
    #plt.plot([f for f in freq_dict.keys()], [cur/cnt for cur, cnt in freq_dict.values()])
    #save the plot as png
    #plt.savefig("freq_plot.png")    