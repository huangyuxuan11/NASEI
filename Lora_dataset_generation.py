import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import roc_curve, auc , confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop


from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram

file_path = 'LoRa_RFF/dataset/Train/dataset_training_aug.h5'
# file_path = 'LoRa_RFF/dataset/Test/dataset_seen_devices.h5'
dev_range = np.arange(0, 30, dtype=int)
pkt_range = np.arange(0, 1000, dtype=int)
snr_range = np.arange(20, 80)
'''
train_feature_extractor trains an RFF extractor using triplet loss.

INPUT: 
    FILE_PATH is the path of training dataset.

    DEV_RANGE is the label range of LoRa devices to train the RFF extractor.

    PKT_RANGE is the range of packets from each LoRa device to train the RFF extractor.

    SNR_RANGE is the SNR range used in data augmentation. 

RETURN:
    FEATURE_EXTRACTOR is the RFF extractor which can extract features from
    channel-independent spectrograms.
'''

LoadDatasetObj = LoadDataset()

# Load preamble IQ samples and labels.
data, label = LoadDatasetObj.load_iq_samples(file_path, dev_range, pkt_range)

# Add additive Gaussian noise to the IQ samples.
data = awgn(data, snr_range)

ChannelIndSpectrogramObj = ChannelIndSpectrogram()

# Convert time-domain IQ samples to channel-independent spectrograms.
data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)
np.save('LoRa_RFF/dataset/Train/X_train_30Class.npy', data)
np.save('LoRa_RFF/dataset/Train/Y_train_30Class.npy', label)

