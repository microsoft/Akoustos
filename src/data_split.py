#!/usr/bin/env python3
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

""" spectrogram.py: Utilities for data pre-processing of binary classification
"""

import pandas as pd
import glob
from sklearn.model_selection import train_test_split


class Data_Split:
    def data_split(categories, spectrogram_dir, train_size = 0.7, val_size = 0.15, test_size = 0.15, by = 'random', include_no_label_category = True):
        
        if sum([train_size, val_size, test_size]) != 1:
            raise ValueError("The total percentage of training/validation/testing datasets need to be 100%.")
        
        spectrogram_filenames = sorted(glob.glob(spectrogram_dir + '*.png'))
        if len(spectrogram_filenames) == 0 or spectrogram_filenames is None:
            print("There are no spectrograms in the pectograms directory: {0}".format(spectrogram_dir))
            return None;
        
        if include_no_label_category == False:
            spectrogram_filenames = [filename for filename in spectrogram_filenames if 'No Label' not in filename]
        
        df = pd.DataFrame()
        df['filename'] = spectrogram_filenames
        
        if len(categories) == 1:
            df['label'] = [1 if filename.replace('.png', '').split('-')[-1] == categories[0] else 0 for filename in spectrogram_filenames]
        elif len(categories) > 1:
            df['label'] = [filename.replace('.png', '').split('-')[-1] if filename.replace('.png', '').split('-')[-1] in categories else '-9999' for filename in spectrogram_filenames]
        else:
            df['label'] = [filename.replace('.png', '').split('-')[-1] for filename in spectrogram_filenames]
        
        if by == 'random':
            df_index = list(range(len(df)))
            index_train_val, index_test = train_test_split(df_index, test_size = test_size, random_state = 1)
            index_train, index_val = train_test_split(index_train_val, test_size = val_size / (1 - test_size), random_state = 1)
            df['split'] = [x[1] for x in sorted(list(zip(index_train + index_val + index_test, 
                                                         ['train'] * len(index_train) + ['val'] * len(index_val) + ['test'] * len(index_test))), 
                                                key = lambda x:x[0])
                          ]
        elif by == 'order': ## if the audio filename has info of audio recorded time, then the split is based on time
            df['split'] = [train] * floor(len(df) * train_size) + ['val'] * floor(len(df) * val_size) + ['test'] * (len(df) - floor(len(df) * train_size) - floor(len(df) * val_size))
            
        summary = df.groupby(['label']).size().reset_index(name='Count')
        summary['Percentage'] = round(100 * summary['Count']  / summary['Count'].sum(), 2)
        
        print(summary)
        print('Size of train, val, test dataset:', 
              len(df.loc[df.split == 'train']), 
              len(df.loc[df.split == 'val']), 
              len(df.loc[df.split == 'test'])
             )
        print('Percentage of train, val, test dataset:', 
              "%.1f" % (100 * len(df.loc[df.split == 'train']) / len(df)) + '%', 
              "%.1f" % (100 * len(df.loc[df.split == 'val']) / len(df)) + '%', 
              "%.1f" % (100 * len(df.loc[df.split == 'test']) / len(df)) + '%'
             )
        
        return df
    