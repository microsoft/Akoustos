#!/usr/bin/env python3
""" spectrogram.py: Utilities for dealing with spectrograms
"""

from datetime import datetime
from scipy import signal
import numpy as np
import os
from math import floor
from joblib import Parallel, delayed
import multiprocessing
import glob


from src.audio import Audio


class Spectrogram:
    
    def clear_space(directory):
        files = glob.glob(directory + '*')
        for f in files:
            os.remove(f)
    
    def generate_spectrograms(spectrogram_duration, labeled_data, audio_filenames, save_to_dir, 
                              axis = False, melspectrogram = False, cmap = 'viridis'):
        
        if not isinstance(spectrogram_duration, int) or spectrogram_duration <= 0:
            raise ValueError("spectrogram_duration needs to be positive integer.")
        
        #labeled_data = Load_Data.labeled_data(directory = labeled_data_dir)
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
                    
                    
                    

    def generate_spectrograms_parallel(spectrogram_duration, labeled_data, audio_filenames, save_to_dir, 
                              axis = False, melspectrogram = False, cmap = 'viridis'):

        begin = datetime.now()
        
        if not isinstance(spectrogram_duration, int) or spectrogram_duration <= 0:
            raise ValueError("spectrogram_duration needs to be positive integer.")
        
        #labeled_data = Load_Data.labeled_data(directory = labeled_data_dir)
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
        
        end = datetime.now()
        print('Time spent to generate spectrograms with parallelization: ', (end - begin).total_seconds(), 'seconds')
        print('Total number of spectrograms produced:', len(glob.glob(save_to_dir + '*')))

        
        
        
        
        
        

    