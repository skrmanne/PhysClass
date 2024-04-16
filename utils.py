import numpy as np
import scipy
import scipy.signal
import cv2
import h5py

from scipy.sparse import spdiags

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _calculate_fft_hr(ppg_signal, fs=30, low_pass=0.5, high_pass=5.0):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    #print(ppg_signal[::10])
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])

    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)

    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0]
    return fft_hr

def get_signal(scores):
    # run through scores and add a constant value if score==1 else subtract
    signal, eps = [0.0], 0.1
    for score in scores[1:]:
        signal.append(signal[-1] + eps if score >= 0.5 else signal[-1] - eps)
    
    signal = np.array(signal)
    return (signal-np.min(signal))/(np.max(signal)-np.min(signal))

def get_classes(signal):
    # run through the signal and append 1.0 if signal[i] > signal[i-1] else 0.0
    classes = [0]
    for i in range(1, len(signal)):
        classes.append(1 if signal[i] > signal[i-1] else 0)
    
    return np.array(classes)

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal

# read video
def read_video(video_file):
    """Reads a video file, returns frames(T,H,W,3) """
    VidObj = cv2.VideoCapture(video_file)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, frame = VidObj.read()
    frames = list()

    while (success):
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)/255.0 # Normalization
        frame[np.isnan(frame)] = 0  # TODO: maybe change into avg
        frames.append(frame)
        success, frame = VidObj.read()

    return np.asarray(frames)

# read signal
def read_wave(wave_file):
    """Reads the label file."""
    f = h5py.File(wave_file, 'r')
    angle = f["angle"][:]
    freq = f["freq"][:]

    # normalize angle signal
    angle = (angle-np.min(angle))/(np.max(angle)-np.min(angle))
    return angle, freq
