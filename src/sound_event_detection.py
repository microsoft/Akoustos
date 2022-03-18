'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

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

from audio import Audio
from preprocessing import speechproc
from preprocessing import spectrogating
from copy import deepcopy
from scipy.signal import lfilter
    
class Sound_Event_Detection:

    def __init__(self, labeled_data_filenames, audio_filenames):
        self.labeled_data = labeled_data_filenames
        self.audio_filenames = audio_filenames
        #Set to No Sound Event as default. The first time it runs will be in the data pre-processing phase.
        self.category = 'No Sound Event'

    def sound_event_detection_for_single_audio_file(self, annotation_base_audio_filename):
        df = pd.DataFrame(columns = list(self.labeled_data))
        matching_audio_filename = [audio_filename for audio_filename in self.audio_filenames if os.path.basename(audio_filename) == annotation_base_audio_filename]
        if len(matching_audio_filename) > 0:
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
                        'Category': self.category}
                df = df.append(new_row, ignore_index=True)
            return df
        else:
            print("no matching single audio file for name: {}".format(annotation_base_audio_filename))    

    def sound_event_detection_for_all_audio_files(self, is_preprocessing_phase=True):
        if is_preprocessing_phase:
            self.sound_event_category = 'No Sound Event'
        else:
            self.sound_event_category = 'Sound Event'

        begin = datetime.now()
        num_cores = multiprocessing.cpu_count()
        with Pool(processes=num_cores) as pool:
            print("Running sound event detection for all annotated data...")
            annotation_base_audio_filenames = list(self.labeled_data['Begin File'].unique())
            df_list = pool.map(self.sound_event_detection_for_single_audio_file, annotation_base_audio_filenames)
            try:
                no_sound_event_data = pd.concat(df_list, ignore_index=True)
            except ValueError:
                no_sound_event_data = None

        print('done with sound event detection:')        
            
        end = datetime.now()
        print('Time spent to preprocess data: ', (end - begin).total_seconds(), 'seconds')
        return no_sound_event_data