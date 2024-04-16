# imports
import os, sys
import random
import numpy as np
import torch
from torchvision import datasets, transforms, io
import h5py
from tqdm import tqdm

# constants for dataset creation
_FPS = 30
_SEC = 5
_LO = 0.5
_HI = 5.0
_NUM_FRAMES = _FPS * _SEC
_PER_CLASS = 100
_NUM_CLASS = 10

# load MNIST dataset from pytorch datasets
mnist_trainset = datasets.MNIST(root='./data', train=True, download=False, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=False, transform=None)

print("MNIST train len:", len(mnist_trainset))
print("MNIST test len:", len(mnist_testset))

# iterate through the train dataset per class and get the images
for i in range(_NUM_CLASS):
    os.makedirs("./data/train/" + str(i), exist_ok=True)
    os.makedirs("./data/test/" + str(i), exist_ok=True)

def generate_dataset(dataset, split):
    freqs = np.random.uniform(_LO, _HI, size=_PER_CLASS*_NUM_CLASS)
    
    for i in tqdm(range(_NUM_CLASS)):
        idx = np.where(dataset.targets == i)[0]
        idx = np.random.choice(idx, _PER_CLASS, replace=False)

        for j in range(_PER_CLASS):
            img = dataset.data[idx[j]]
            label = dataset.targets[idx[j]]
            
            # bug for rotate only accepts 3 dimension images
            img = img.unsqueeze(0)
            
            # sample random frequency
            freq =  freqs[i*_PER_CLASS + j]

            # random offset to start the rotation.
            offset = np.random.uniform(-np.pi/2, np.pi/2)

            # create _NUM angles to rotate the image at a random speed
            angles = [90. * np.sin(2*np.pi*freq*t/_FPS + offset) for t in range(_NUM_FRAMES)]
            vid = []

            for angle in angles:
                img_rot = transforms.functional.rotate(img, angle)
                # append img_rot in channel dimension to make it a 3 channel image - bug for io.write_video: only accepts 3 channel images
                img_rot = img_rot.permute(1,2,0)
                img_rot = torch.cat((img_rot, img_rot, img_rot), dim=2)
                vid.append(img_rot)
            
            # create a video file from vid and write it as an avi file
            vid = torch.stack(vid)
            fname = "./data/" + split + "/" + str(i) + "/" + str(idx[j])
            io.write_video(fname + ".mp4", vid, _FPS)
            #print("Writing video:", fname)

            # write the angles as radians and freqs to an hdf5 file
            hf = h5py.File(fname+'.h5', 'w')
            hf.create_dataset('angle', data=[np.pi*angle/180. for angle in angles])
            hf.create_dataset('freq', data=[freq])
            hf.close()

# generate train and testsets
generate_dataset(mnist_trainset, "train")
generate_dataset(mnist_testset, "test")