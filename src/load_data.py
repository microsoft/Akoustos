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
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool


from src.audio import Audio
from src.preprocessing import speechproc
from src.preprocessing import spectrogating
from copy import deepcopy
from scipy.signal import lfilter
    

class Load_Data:
    
    def audio_filenames(directory):
        begin = datetime.now()
        
        audio_filenames = glob.glob(directory + '*')
        
        end = datetime.now()
        print('Number of audio files:', len(audio_filenames))  
        print('Time spent to upload audio files: ', (end - begin).total_seconds(), 'seconds')
        
        return audio_filenames


    def labeled_data(labeled_data_dir, audio_dir):
        begin = datetime.now()
        
        filenames = glob.glob(labeled_data_dir + '*')
        all_labeled_data = pd.DataFrame()
        for filename in filenames:
            if filename.endswith('xlsx'):
                labeled_data = pd.read_excel(filename)
            elif filename.endswith('csv'):
                labeled_data = pd.read_csv(filename)
            all_labeled_data = all_labeled_data.append(labeled_data, ignore_index=True)
        all_labeled_data = all_labeled_data.sort_values(by=['Begin File', 'Begin Time (s)']).reset_index(drop=True)
        
        
        summary = all_labeled_data.groupby(['Category']).size().reset_index(name='Count')
        summary['Percentage'] = round(100 * summary['Count']  / summary['Count'].sum(), 2)
        print(summary)
        
        end = datetime.now()
        print('Time spent to upload labeled data: ', (end - begin).total_seconds(), 'seconds')
        
        return all_labeled_data
        
        

    def labeled_data_with_NoSoundEvent(labeled_data_dir, audio_dir):
        begin = datetime.now()
        
        filenames = glob.glob(labeled_data_dir + '*')
        all_labeled_data = pd.DataFrame()
        for filename in filenames:
            if filename.endswith('xlsx'):
                labeled_data = pd.read_excel(filename)
            elif filename.endswith('csv'):
                labeled_data = pd.read_csv(filename)
            all_labeled_data = all_labeled_data.append(labeled_data, ignore_index=True)
        all_labeled_data = all_labeled_data.sort_values(by=['Begin File', 'Begin Time (s)']).reset_index(drop=True)
        
        ### no_labeled_data
        #no_labeled_data = pd.DataFrame(columns = list(labeled_data))
        #for i in range(0, len(all_labeled_data) - 1, len(all_labeled_data.Category.unique())):
        #    if all_labeled_data.loc[i, 'Begin File'] != all_labeled_data.loc[i + 1, 'Begin File']:
        #        new_row = {'Begin Time (s)': 0,
        #                   'End Time (s)': floor(all_labeled_data.loc[i, 'Begin Time (s)']),
        #                   'Low Freq (Hz)': 0,
        #                   'High Freq (Hz)': 0,
        #                   'Begin File': all_labeled_data.loc[i, 'Begin File'],
        #                   'Category': 'No Label'}
        #    elif all_labeled_data.loc[i, 'Begin File'] == all_labeled_data.loc[i + 1, 'Begin File']:
        #        new_row = {'Begin Time (s)': ceil(all_labeled_data.loc[i, 'End Time (s)']),
        #                   'End Time (s)': floor(all_labeled_data.loc[i + 1, 'Begin Time (s)']),
        #                   'Low Freq (Hz)': 0,
        #                   'High Freq (Hz)': 0,
        #                   'Begin File': all_labeled_data.loc[i, 'Begin File'],
        #                   'Category': 'No Label'}
        #    no_labeled_data = no_labeled_data.append(new_row, ignore_index=True)
        #all_labeled_data = all_labeled_data.append(no_labeled_data, ignore_index=True)
        
        
        ### no_sound_event_data
        no_sound_event_data = pd.DataFrame(columns = list(all_labeled_data))
        annotation_base_audio_filenames = list(all_labeled_data['Begin File'].unique())
        audio_filenames = glob.glob(audio_dir + '*')
        
        for i, annotation_base_audio_filename in enumerate(annotation_base_audio_filenames):
            matching_audio_filename = [audio_filename for audio_filename in audio_filenames if os.path.basename(audio_filename) == annotation_base_audio_filename]
            audio = Audio.load(matching_audio_filename.pop())
    
            ## sound event detection
            noise = audio.samples[0:1*audio.sample_rate]
            x_dn = spectrogating.removeNoise(audio_clip=audio.samples, 
                                             noise_clip=noise,
                                             n_grad_freq=2,
                                             n_grad_time=4,
                                             n_fft=2048,
                                             win_length=2048,
                                             hop_length=512,
                                             n_std_thresh=2.5,
                                             prop_decrease=1.0,
                                             verbose=False,
                                             visual=False)

            winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 2048
            ftThres = 0.4
            vadThres = 0.2
            opts = 1
            ft, flen, fsh10, nfr10 = speechproc.sflux(x_dn, audio.sample_rate, winlen, ovrlen, nftt)
            # --spectral flatness --
            pv01 = np.zeros(nfr10)
            pv01[np.less_equal(ft, ftThres)] = 1 
            pitch = deepcopy(ft)
            pvblk = speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)
            # --filtering--
            ENERGYFLOOR = np.exp(-50)
            b = np.array([0.9770,   -0.9770])
            a = np.array([0.3,   -0.3])
            fdata = lfilter(b, a, x_dn, axis=0)
            vad_seg = speechproc.snre_vad(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres)  
            no_events_starttime = [0] + [i / len(vad_seg) * audio.duration() for i in range(len(vad_seg)) if vad_seg[i] == 0 and vad_seg[i-1] == 1]
            no_events_endtime = [i / len(vad_seg) * audio.duration() for i in range(len(vad_seg)) if vad_seg[i] == 1 and vad_seg[i-1] == 0] + [audio.duration()]
            for start, end in zip(no_events_starttime, no_events_endtime):
                new_row = {'Begin Time (s)': start,
                           'End Time (s)': end,
                           'Low Freq (Hz)': 0,
                           'High Freq (Hz)': 0,
                           'Begin File': annotation_base_audio_filename,
                           'Category': 'No Sound Event'}
                no_sound_event_data = no_sound_event_data.append(new_row, ignore_index=True)
                
            no_sound_event_data['duration'] = no_sound_event_data['End Time (s)'] - no_sound_event_data['Begin Time (s)'] 
            no_sound_event_data = no_sound_event_data.sort_values(by='duration',ascending=False)[: len(all_labeled_data) // len(all_labeled_data.Category.unique())]
            no_sound_event_data = no_sound_event_data.drop(['duration'], axis = 1)
                
        
        all_labeled_data = all_labeled_data.append(no_sound_event_data, ignore_index=True)
        summary = all_labeled_data.groupby(['Category']).size().reset_index(name='Count')
        summary['Percentage'] = round(100 * summary['Count']  / summary['Count'].sum(), 2)
        print(summary)
        
        end = datetime.now()
        print('Time spent to preprocess data: ', (end - begin).total_seconds(), 'seconds')
        
        return all_labeled_data

    
    
    
    
    
        
    def labeled_data_with_NoSoundEvent_parallel(labeled_data_dir, audio_dir):
        ###################
         ### TODO: fix
        ###################
        
        begin = datetime.now()
        
        filenames = glob.glob(labeled_data_dir + '*')
        all_labeled_data = pd.DataFrame()
        for filename in filenames:
            if filename.endswith('xlsx'):
                labeled_data = pd.read_excel(filename)
            elif filename.endswith('csv'):
                labeled_data = pd.read_csv(filename)
            all_labeled_data = all_labeled_data.append(labeled_data, ignore_index=True)
        all_labeled_data = all_labeled_data.sort_values(by=['Begin File', 'Begin Time (s)']).reset_index(drop=True)
        
        ### no_labeled_data
        #no_labeled_data = pd.DataFrame(columns = list(labeled_data))
        #for i in range(0, len(all_labeled_data) - 1, len(all_labeled_data.Category.unique())):
        #    if all_labeled_data.loc[i, 'Begin File'] != all_labeled_data.loc[i + 1, 'Begin File']:
        #        new_row = {'Begin Time (s)': 0,
        #                   'End Time (s)': floor(all_labeled_data.loc[i, 'Begin Time (s)']),
        #                   'Low Freq (Hz)': 0,
        #                   'High Freq (Hz)': 0,
        #                   'Begin File': all_labeled_data.loc[i, 'Begin File'],
        #                   'Category': 'No Label'}
        #    elif all_labeled_data.loc[i, 'Begin File'] == all_labeled_data.loc[i + 1, 'Begin File']:
        #        new_row = {'Begin Time (s)': ceil(all_labeled_data.loc[i, 'End Time (s)']),
        #                   'End Time (s)': floor(all_labeled_data.loc[i + 1, 'Begin Time (s)']),
        #                   'Low Freq (Hz)': 0,
        #                   'High Freq (Hz)': 0,
        #                   'Begin File': all_labeled_data.loc[i, 'Begin File'],
        #                   'Category': 'No Label'}
        #    no_labeled_data = no_labeled_data.append(new_row, ignore_index=True)
        #all_labeled_data = all_labeled_data.append(no_labeled_data, ignore_index=True)
        
        
        ### no_sound_event_data
        annotation_base_audio_filenames = list(all_labeled_data['Begin File'].unique())
        audio_filenames = glob.glob(audio_dir + '*')

        def sound_event_detection_for_single_audio_file(annotation_base_audio_filename):
            df = pd.DataFrame(columns = list(all_labeled_data))
            matching_audio_filename = [audio_filename for audio_filename in audio_filenames if os.path.basename(audio_filename) == annotation_base_audio_filename]
            audio = Audio.load(matching_audio_filename.pop())
    
            ## sound event detection
            noise = audio.samples[0:1*audio.sample_rate]
            x_dn = spectrogating.removeNoise(audio_clip=audio.samples, 
                                             noise_clip=noise,
                                             n_grad_freq=2,
                                             n_grad_time=4,
                                             n_fft=2048,
                                             win_length=2048,
                                             hop_length=512,
                                             n_std_thresh=2.5,
                                             prop_decrease=1.0,
                                             verbose=False,
                                             visual=False)

            winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 2048
            ftThres = 0.4
            vadThres = 0.2
            opts = 1
            ft, flen, fsh10, nfr10 = speechproc.sflux(x_dn, audio.sample_rate, winlen, ovrlen, nftt)
            # --spectral flatness --
            pv01 = np.zeros(nfr10)
            pv01[np.less_equal(ft, ftThres)] = 1 
            pitch = deepcopy(ft)
            pvblk = speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)
            # --filtering--
            ENERGYFLOOR = np.exp(-50)
            b = np.array([0.9770,   -0.9770])
            a = np.array([0.3,   -0.3])
            fdata = lfilter(b, a, x_dn, axis=0)
            vad_seg = speechproc.snre_vad(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres)  
            no_events_starttime = [0] + [i / len(vad_seg) * audio.duration() for i in range(len(vad_seg)) if vad_seg[i] == 0 and vad_seg[i-1] == 1]
            no_events_endtime = [i / len(vad_seg) * audio.duration() for i in range(len(vad_seg)) if vad_seg[i] == 1 and vad_seg[i-1] == 0] + [audio.duration()]
            for start, end in zip(no_events_starttime, no_events_endtime):
                new_row = {'Begin Time (s)': start,
                           'End Time (s)': end,
                           'Low Freq (Hz)': 0,
                           'High Freq (Hz)': 0,
                           'Begin File': annotation_base_audio_filename,
                           'Category': 'No Sound Event'}
                df = df.append(new_row, ignore_index=True)
            return df
            
                
            
        num_cores = multiprocessing.cpu_count()
        with Pool(processes=num_cores) as pool:
            df_list = pool.map(sound_event_detection_for_single_audio_file, annotation_base_audio_filenames)
            no_sound_event_data = pd.concat(df_list, ignore_index=True)

        no_sound_event_data['duration'] = no_sound_event_data['End Time (s)'] - no_sound_event_data['Begin Time (s)'] 
        no_sound_event_data = no_sound_event_data.sort_values(by='duration',ascending=False)[: len(all_labeled_data) // len(all_labeled_data.Category.unique())]
        no_sound_event_data = no_sound_event_data.drop(['duration'], axis = 1)
 
    
        all_labeled_data = all_labeled_data.append(no_sound_event_data, ignore_index=True)
        summary = all_labeled_data.groupby(['Category']).size().reset_index(name='Count')
        summary['Percentage'] = round(100 * summary['Count']  / summary['Count'].sum(), 2)
        print(summary)
        
        end = datetime.now()
        print('Time spent to preprocess data: ', (end - begin).total_seconds(), 'seconds')
        
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
    