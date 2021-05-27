#!/usr/bin/env python3
""" spectrogram.py: Utilities for dealing with spectrograms
"""

from scipy import signal
import numpy as np
import os
from math import floor
from joblib import Parallel, delayed
import multiprocessing
import cv2
import glob
from statistics import median

from src.audio import Audio
from src.labeled_data import Labeled_Data


class Spectrogram:
    """ 
    ## TODO: clear all contents in save_to_dir before generating (mel)-spectrograms
    """ 
    def generate_spectrograms(spectrogram_duration, labeled_data_filenames: list, audio_filenames: list, save_to_dir, 
                              axis = False, melspectrogram = False, cmap = 'viridis'):
        
        if not isinstance(spectrogram_duration, int) or spectrogram_duration <= 0:
            raise ValueError("spectrogram_duration needs to be positive integer.")
        
        labeled_data = Labeled_Data.load(filenames = labeled_data_filenames)
        audio_filenames_base  = [os.path.basename(audio_filename) for audio_filename in audio_filenames]
        
        if not set(labeled_data['Begin File']).intersection(set(audio_filenames_base)):
            raise Exception("No matching audio files.")
        
        else:
            for index, row in labeled_data.iterrows():
                annotation_base_audio_filename = row['Begin File']
                matching_audio_filename = [audio_filename for audio_filename in audio_filenames if os.path.basename(audio_filename) == annotation_base_audio_filename]
                if not matching_audio_filename:
                    continue
                else:
                    matching_audio_filename = matching_audio_filename.pop()
                    audio = Audio.load(matching_audio_filename)
                    audio_duration = audio.duration()
                    start_time = row['Begin Time (s)']
                    end_time = row['End Time (s)']
                    category = row['Category']
                    if floor(start_time) + spectrogram_duration <= audio_duration:
                        spectrogram_start_time = floor(start_time)
                        spectrogram_end_time = spectrogram_start_time + spectrogram_duration
                    else:
                        spectrogram_end_time = audio_duration
                        spectrogram_start_time = audio_duration - spectrogram_duration
                    audio_trim = audio.trim(start_time = spectrogram_start_time, end_time = spectrogram_end_time)
                    audio_trim.generate_spectrogram(axis = axis, melspectrogram = melspectrogram, cmap = cmap, 
                                                    filename = save_to_dir + '_'.join([str(x) for x in [annotation_base_audio_filename,spectrogram_start_time, spectrogram_end_time, category]]) + '.png')
                    
                    
                    

    def generate_spectrograms_parallel(spectrogram_duration, labeled_data_filenames: list, audio_filenames: list, save_to_dir, 
                              axis = False, melspectrogram = False, cmap = 'viridis'):

        if not isinstance(spectrogram_duration, int) or spectrogram_duration <= 0:
            raise ValueError("spectrogram_duration needs to be positive integer.")
        
        labeled_data = Labeled_Data.load(filenames = labeled_data_filenames)
        audio_filenames_base  = [os.path.basename(audio_filename) for audio_filename in audio_filenames]
        
        if not set(labeled_data['Begin File']).intersection(set(audio_filenames_base)):
            raise Exception("No matching audio files.")
        
        
        def generate_single_spectrogram(spectrogram_duration, row, save_to_dir):
            annotation_base_audio_filename = row['Begin File']
            matching_audio_filename = [audio_filename for audio_filename in audio_filenames if os.path.basename(audio_filename) == annotation_base_audio_filename]
            if  matching_audio_filename:
                matching_audio_filename = matching_audio_filename.pop()
                audio = Audio.load(matching_audio_filename)
                audio_duration = floor(audio.duration())
                start_time = row['Begin Time (s)']
                end_time = row['End Time (s)']
                category = row['Category']
                if floor(start_time) + spectrogram_duration <= audio_duration:
                    spectrogram_start_time = floor(start_time)
                    spectrogram_end_time = spectrogram_start_time + spectrogram_duration
                else:
                    spectrogram_end_time = audio_duration
                    spectrogram_start_time = audio_duration - spectrogram_duration
                audio_trim = audio.trim(start_time = spectrogram_start_time, end_time = spectrogram_end_time)
                audio_trim.generate_spectrogram(axis = axis, melspectrogram = melspectrogram, cmap = cmap, 
                                                filename = save_to_dir + '_'.join([str(x) for x in [annotation_base_audio_filename,spectrogram_start_time, spectrogram_end_time, category]]) + '.png')
        
        
        def generate_spectrograms_by_row(row):
            annotation_base_audio_filename = row['Begin File']
            try:
                return generate_single_spectrogram(spectrogram_duration, row, save_to_dir)
            except:
                pass
        num_cores = multiprocessing.cpu_count()
        spectrograms = Parallel(n_jobs=num_cores)(delayed(generate_spectrograms_by_row)(row) for index, row in labeled_data.iterrows())



    def load_spectrograms(spectrogram_dir, shape=(224, 224)):
        """
        load spectrograms into vector

        Args:
            filename: path of image to load
            shape: tuple of (nrow, ncol)

        """
        filenames = glob.glob(spectrogram_dir + '*.png')
        spectrogram_median_file_size = median([os.path.getsize(filename) for filename in filenames])
        spectrogram_vector = []
        spectrogram_filenames = []
        for filename in filenames:
            if os.path.getsize(filename) >= spectrogram_median_file_size * 0.8:
                img = cv2.imread(filename)  
                img = cv2.resize(img, shape) / 255.0
                spectrogram_vector.append(img)
                spectrogram_filenames.append(filename)
        return np.asarray(spectrogram_vector), spectrogram_filenames
    
    