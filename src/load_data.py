#!/usr/bin/env python3
""" labeled_data.py: Utilities for dealing with labeled data
"""

from datetime import datetime
import glob
import pandas as pd
import numpy as np
import os
from statistics import median
import cv2
import warnings
from math import ceil, floor

    
class Load_Data:
    
    def audio_filenames(directory):
        audio_filenames = glob.glob(directory + '*')
        end = datetime.now()
        print('Number of audio files:', len(audio_filenames))  
        
        return audio_filenames


    def labeled_data(directory):
        """ 
        TODO: handle exceptions
        """
        filenames = glob.glob(directory + '*')
        all_labeled_data = pd.DataFrame()
        for filename in filenames:
            if filename.endswith('xlsx'):
                labeled_data = pd.read_excel(filename)
            elif filename.endswith('csv'):
                labeled_data = pd.read_csv(filename)
            all_labeled_data = all_labeled_data.append(labeled_data, ignore_index=True)
        all_labeled_data = all_labeled_data.sort_values(by=['Begin File', 'Begin Time (s)']).reset_index(drop=True)
        
        no_labeled_data = pd.DataFrame(columns = list(labeled_data))
        for i in range(0, len(all_labeled_data) - 1, len(all_labeled_data.Category.unique())):
            if all_labeled_data.loc[i, 'Begin File'] != all_labeled_data.loc[i + 1, 'Begin File']:
                new_row = {'Begin Time (s)': 0,
                           'End Time (s)': floor(all_labeled_data.loc[i, 'Begin Time (s)']),
                           'Low Freq (Hz)': 0,
                           'High Freq (Hz)': 0,
                           'Begin File': all_labeled_data.loc[i, 'Begin File'],
                           'Category': 'No Label'}
            elif all_labeled_data.loc[i, 'Begin File'] == all_labeled_data.loc[i + 1, 'Begin File']:
                new_row = {'Begin Time (s)': ceil(all_labeled_data.loc[i, 'End Time (s)']),
                           'End Time (s)': floor(all_labeled_data.loc[i + 1, 'Begin Time (s)']),
                           'Low Freq (Hz)': 0,
                           'High Freq (Hz)': 0,
                           'Begin File': all_labeled_data.loc[i, 'Begin File'],
                           'Category': 'No Label'}
            no_labeled_data = no_labeled_data.append(new_row, ignore_index=True)
        
        all_labeled_data = all_labeled_data.append(no_labeled_data, ignore_index=True)
        
        summary = all_labeled_data.groupby(['Category']).size().reset_index(name='Count')
        summary['Percentage'] = round(100 * summary['Count']  / summary['Count'].sum(), 2)
        print(summary)
        
        return all_labeled_data


    def load_spectrograms(directory, shape=(224, 224)):
        """
        load spectrograms into vector

        Args:
            filename: path of image to load
            shape: tuple of (nrow, ncol)

        """
        begin = datetime.now()
        
        spectrogram_filenames = glob.glob(directory + '*.png')
        spectrogram_median_file_size = median([os.path.getsize(filename) for filename in spectrogram_filenames])
        spectrogram_vector = []
        for filename in spectrogram_filenames:
            if os.path.getsize(filename) >= spectrogram_median_file_size * 0.8:
                img = cv2.imread(filename)  
                img = cv2.resize(img, shape) / 255.0
                spectrogram_vector.append(img)
        
        end = datetime.now()
        print('number of valid spectrograms:', len(spectrogram_filenames))
        print('shape of vector for valid spectrograms:', spectrogram_vector[0].shape)
        print('Time spent to load spectrograms as array: ', (end - begin).total_seconds(), 'seconds')

        return np.asarray(spectrogram_vector), spectrogram_filenames
    