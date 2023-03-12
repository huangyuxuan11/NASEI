import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

from sklearn.model_selection import train_test_split



from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram
from deep_learning_models import TripletNet, identity_loss

file_path = 'LoRa_RFF/dataset/Train/dataset_training_aug.h5'
# file_path = 'LoRa_RFF/dataset/Test/dataset_seen_devices.h5'
dev_range = np.arange(0, 30, dtype=int)
pkt_range = np.arange(0, 1000, dtype=int)
# snr_range = np.arange(20, 80)
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
# data = awgn(data, snr_range)

ChannelIndSpectrogramObj = ChannelIndSpectrogram()

# Convert time-domain IQ samples to channel-independent spectrograms.
# data = ChannelIndSpectrogramObj.normalization(data)

data_I = np.real(data)
data_Q = np.imag(data)

print(data_I.shape)
data_I = data_I.reshape(-1, 8192, 1)
data_Q = data_Q.reshape(-1, 8192, 1)
data = np.concatenate((data_I, data_Q), axis=2)
print(data.shape)
print(label.shape)
print(data[0])
np.save('/data1/huangyx/ADS-B_DML/Dataset/X_lora_train_30Class.npy', data)
np.save('/data1/huangyx/ADS-B_DML/Dataset/Y_lora_train_30Class.npy', label)
print('save successfully')



#
# # Create an RFF extractor.
# feature_extractor = TripletNetObj.feature_extractor(data.shape)
#
# # Create the Triplet net using the RFF extractor.
# triplet_net = TripletNetObj.create_triplet_net(feature_extractor, margin)
#
# # Create callbacks during training. The training stops when validation loss
# # does not decrease for 30 epochs.
# early_stop = EarlyStopping('val_loss',
# min_delta = 0,
# patience =
# patience)
#
# reduce_lr = ReduceLROnPlateau('val_loss',
# min_delta = 0,
# factor = 0.2,
# patience = 10,
# verbose = 1)
# callbacks = [early_stop, reduce_lr]

# Split the dasetset into validation and training sets.

