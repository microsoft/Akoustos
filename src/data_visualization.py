#!/usr/bin/env python3

'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

from load_data import Load_Data
import matplotlib.pyplot as plt
from audio import Audio
import os
import numpy as np
import librosa

class Data_Visualization:
    
    def histogram_call_duration(labeled_data_dir, audio_dir):
        labeled_data = Load_Data.labeled_data(labeled_data_dir)
        if labeled_data is not None:
            plt.hist(labeled_data['End Time (s)'] - labeled_data['Begin Time (s)'])
    

    def sample_spectrograms(labeled_data_dir, audio_dir, length_in_seconds, cmap = 'viridis'):
        ### choice of y_axis: 'linear', 'mel', 'log'
        audio_filenames = Load_Data.audio_filenames(directory = audio_dir)
        labeled_data = Load_Data.labeled_data(labeled_data_dir)
        if labeled_data is None:
            print("There is no input data to generate spectrograms.")
            return
        
        category_list = labeled_data.Category.unique().tolist()
        for i, category in enumerate(category_list):
            detection = labeled_data.loc[labeled_data.Category == category].iloc[0]
            matching_audio_filename = [audio_filename for audio_filename in audio_filenames if os.path.basename(audio_filename) == detection['Begin File']]
            # check if there is any matching audio filename
            if len(matching_audio_filename) == 0 or matching_audio_filename is None:
                print("No matching audio was found while trying to get sample spectrograms.")
                return

            audio = Audio.load(matching_audio_filename.pop())
            audio_samples, sample_rate = audio.samples, audio.sample_rate

            spectrogram_start_time = int(detection['Begin Time (s)'])
            spectrogram_end_time = min(audio.duration(), spectrogram_start_time + length_in_seconds)
            samples_trimmed =audio_samples[sample_rate*spectrogram_start_time:sample_rate*spectrogram_end_time]
            
            D = librosa.amplitude_to_db(np.abs(librosa.stft(samples_trimmed)), ref=np.max)
            fig=plt.figure(figsize=(16, 20))
            plt.subplot(len(category_list), 3, i * 3 + 1)
            librosa.display.specshow(D, x_axis = 'time', y_axis='linear', cmap=cmap)
            plt.title('Linear-spectrogram for category ' + str(category))
            
            plt.subplot(len(category_list), 3, i * 3 + 2)
            librosa.display.specshow(D, x_axis = 'time', y_axis='log', cmap=cmap)
            plt.title('Log-spectrogram for category ' + str(category))
            
            plt.subplot(len(category_list), 3, i * 3 + 3)
            librosa.display.specshow(D, x_axis = 'time',y_axis='mel', cmap=cmap)
            plt.title('Mel-spectrogram for category ' + str(category))