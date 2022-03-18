#!/usr/bin/env python3
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

""" spectrogram.py: Utilities for dealing with spectrograms
"""

from datetime import datetime
from matplotlib import cm
from scipy import signal
import numpy as np
import pandas as pd
import os
from math import floor, ceil
from joblib import Parallel, delayed
import multiprocessing
import glob
import random

from audio import Audio

class Spectrogram:
    def __init__(self,
                raw_audio_dir, 
                spectrogram_duration, 
                labeled_data, 
                save_to_dir,
                axis=False, 
                sr = 22050, 
                hop_length=512, 
                fmin=None, 
                x_axis='time', 
                y_axis='linear', 
                cmap ='viridis'):

        self.raw_audio_dir = raw_audio_dir
        self.labeled_data = labeled_data
        self.spectrogram_duration = spectrogram_duration
        self.save_to_dir=save_to_dir
        self.axis=axis
        self.sr=sr
        self.hop_length=hop_length
        self.fmin=fmin
        self.x_axis=x_axis
        self.y_axis=y_axis
        self.cmap=cmap

    def clear_space(directory):
        files = glob.glob(directory + '*')
        for f in files:
            os.remove(f)
            
    def generate_spectrograms_parallel(self):
        if not isinstance(self.spectrogram_duration, int) or self.spectrogram_duration <= 0:
            raise ValueError("spectrogram_duration needs to be positive integer.")
        
        ### Process in parallel
        num_cores = multiprocessing.cpu_count()
        begin = datetime.now()

        results = Parallel(n_jobs=num_cores)(delayed(self.generate_spectrograms_by_row)(row) for index, row in self.labeled_data.iterrows())
        
        end = datetime.now()
        print('Time spent to generate spectrograms with parallelization: ', (end - begin).total_seconds(), 'seconds')
        print('Total number of spectrograms produced:', len(glob.glob(self.save_to_dir + '*')))

        
    def generate_spectrograms_by_row(self, row):
        try:
            return self.generate_single_spectrogram(row)
        except:
            pass

        
    def generate_single_spectrogram(self, row):
        annotation_base_audio_filename = row['Begin File']
        audio = Audio.load(self.raw_audio_dir + annotation_base_audio_filename)
        if audio:
            audio_duration = floor(audio.duration())
            start_time = row['Begin Time (s)']
            end_time = row['End Time (s)']
            category = row['Category']
            if floor(start_time) + self.spectrogram_duration <= audio_duration:
                spectrogram_start_time = floor(start_time)
                spectrogram_end_time = spectrogram_start_time + self.spectrogram_duration
            else:
                spectrogram_end_time = audio_duration
                spectrogram_start_time = audio_duration - self.spectrogram_duration
            audio_trim = audio.trim(start_time = spectrogram_start_time, end_time = spectrogram_end_time)
            audio_trim.generate_spectrogram(axis=self.axis, sr = self.sr, hop_length=self.hop_length, fmin=self.fmin, fmax=self.fmin, x_axis=self.x_axis, y_axis=self.y_axis, cmap = self.cmap, filename = self.save_to_dir + '-'.join([str(x) for x in [annotation_base_audio_filename,spectrogram_start_time, spectrogram_end_time, category]]) + '.png')
        else:
            print("Could not load audio for file {0}".format(annotation_base_audio_filename))