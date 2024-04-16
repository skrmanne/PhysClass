# dataloader
import cv2
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset

from utils import get_classes

# dataloader: create a custom dataloader for loading the synthetic data
class SyntheticDataset(Dataset):
    """Synthetic rotating digits dataset.
    csv file and a root directory with images.
    """
    def __init__(self, filename, mode):
        with open(filename) as f:
            self.filenames = f.readlines()
            self.filenames = [x.strip() for x in self.filenames]
        self.mode = mode

    def __len__(self):
        return len(self.filenames)
    
    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T,H,W,3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()

        while (success):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            
            # add random gaussian noise to frame using opencv function and clip pixels outside 0-255 range
            #noise = np.random.rand(frame.shape[0], frame.shape[1], frame.shape[2])*0.1*255
            #frame = np.clip(frame+noise.astype(np.uint8), 0, 255)

            frame = np.asarray(frame)/255.0 # Normalization
            frame[np.isnan(frame)] = 0  # TODO: maybe change into avg
            frames.append(frame)
            success, frame = VidObj.read()

        return np.asarray(frames)

    @staticmethod
    def standardized_data(data):
        """Z-score standardization for video data."""
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data

    @staticmethod
    def standardized_label(label):
        """Z-score standardization for label signal."""
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label

    @staticmethod
    def read_wave(wave_file):
        """Reads the label file."""
        f = h5py.File(wave_file, 'r')
        angle = f["angle"][:]
        freq = f["freq"][:]

        # normalize angle signal
        angle = (angle-np.min(angle))/(np.max(angle)-np.min(angle))

        return angle, freq
    
    def __getitem__(self, idx):
        vid_fname = self.filenames[idx]
        wave_fname = vid_fname.replace(".mp4", ".h5")

        # read video
        vid = self.read_video(vid_fname)
        angle, freq = self.read_wave(wave_fname)

        # generate class labels for classification mode
        if self.mode == "classification":
            angle = get_classes(angle)

        # convert to torch tensors
        vid = torch.from_numpy(np.float32(vid)) # TODO: Check how to convert to tensor.
        angle = torch.from_numpy(np.float32(angle))   # Normalization for angle.
        freq = torch.from_numpy(np.float32(freq))   # Frequency remains as-is.

        # permute video:
        vid = torch.permute(vid, (0,3,1,2)) # (T, H, W, C) -> (T, C, H, W)

        # print and check
        #print("vid:{}, angle:{}, freq:{}".format(vid.shape, angle.shape, freq.item()))
        return vid, angle, freq, vid_fname
