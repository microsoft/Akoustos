'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

import os
import random
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import cv2
import scipy.interpolate as interp
import matplotlib.pyplot as plt

from tqdm import tqdm
from copy import deepcopy

from PIL import Image

from scipy.signal import lfilter

import speechproc
import spectrogating

# %%
############
# Load data
############
x, sr = librosa.load('./5AE9F749.wav', sr=48000)

# %%
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x ,sr=sr, x_axis='time')

# %%
################
# Mel Spectrogram
###############
x_mel = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=64)
x_mel_db = librosa.power_to_db(x_mel, ref=np.max)
plt.figure(figsize=(14, 5))
librosa.display.specshow(x_mel_db, x_axis='time', y_axis='mel')
plt.axis('off')

# %%
############
# PCEN
############
x_pcen = librosa.pcen(x_mel, sr=sr, gain=1.3, hop_length=512,
                      bias=2, power=0.3, time_constant=0.4, eps=1e-06, max_size=1)

plt.figure(figsize=(14, 5))
librosa.display.specshow(x_pcen, x_axis='time', y_axis='mel')
plt.axis('off')

# %%
kernel = np.ones((3, 3), np.float32)/25
dst = cv2.filter2D(x_pcen, -1, kernel)
plt.figure(figsize=(14, 5))
librosa.display.specshow(dst, x_axis='time', y_axis='mel')
plt.axis('off')

# %%
##############
# Denoise with spectral gating
##############
noise2 = x[0:1*sr]
x_dn = spectrogating.removeNoise(audio_clip=x, noise_clip=noise2,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=2.5,
    prop_decrease=1.0,
    verbose=False,
    visual=False)

# %%
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr, x_axis='time');
librosa.display.waveplot(x_dn,sr=sr, x_axis='time');
plt.axis('off')

# %%
x_mel_dn = librosa.feature.melspectrogram(y=x_dn, sr=sr, n_mels=64)
x_mel_dn_db = librosa.power_to_db(x_mel_dn, ref=np.max)
plt.figure(figsize=(14, 5))
librosa.display.specshow(x_mel_dn_db, x_axis='time', y_axis='mel')
plt.axis('off')


# %%
###########
# rVAD segmentation
###########
winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 2048
ftThres = 0.4
vadThres = 0.2
opts = 1

# %%
ft, flen, fsh10, nfr10 = speechproc.sflux(x_dn, sr, winlen, ovrlen, nftt)

# %%
# --spectral flatness --
pv01 = np.zeros(nfr10)
pv01[np.less_equal(ft, ftThres)] = 1 
pitch = deepcopy(ft)
pvblk = speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)

# %%
# --filtering--
ENERGYFLOOR = np.exp(-50)
b = np.array([0.9770,   -0.9770])
a = np.array([0.3,   -0.3])
fdata = lfilter(b, a, x_dn, axis=0)

# %%
#--pass 1--
noise_samp, noise_seg, n_noise_samp=speechproc.snre_highenergy(fdata, nfr10, flen, fsh10,
                                                               ENERGYFLOOR, pv01, pvblk)

# %%
#sets noisy segments to zero
for j in range(n_noise_samp):
    fdata[range(int(noise_samp[j, 0]),  int(noise_samp[j, 1]) + 1)] = 0 

# %%
vad_seg = speechproc.snre_vad(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres)

# %%
plt.figure(figsize=(14, 5))
librosa.display.waveplot(vad_seg.astype('float'), sr=sr, x_axis='time')

# %%
import scipy.interpolate as interp
interp = interp.interp1d(np.arange(vad_seg.size), vad_seg)
vad_seg_st = interp(np.linspace(0, vad_seg.size-1, fdata.size))

# %%
plt.figure(figsize=(14, 5))
librosa.display.waveplot(vad_seg_st.astype('float') * fdata.max(), sr=sr, x_axis='time')
librosa.display.waveplot(fdata, sr=sr, x_axis='time')


no_events_starttime = [i / len(vad_seg) * audio.duration() for i in range(len(vad_seg)) if vad_seg[i] == 0 and vad_seg[i-1] == 1]
