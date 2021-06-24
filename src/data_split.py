#!/usr/bin/env python3
""" spectrogram.py: Utilities for data pre-processing of binary classification
"""

import pandas as pd
import glob
from sklearn.model_selection import train_test_split


class Data_Split:
    """ 
    ## TODO: Other ways of spliting
    """ 
    def data_split_binary(category, spectrogram_dir, train_size = 0.7, val_size = 0.15, test_size = 0.15):
        
        if sum([train_size, val_size, test_size]) != 1:
            raise ValueError("The total percentage of training/validation/testing datasets need to be 100%.")
        
        spectrogram_filenames = glob.glob(spectrogram_dir + '*.png')
        
        df = pd.DataFrame()
        df['filename'] = spectrogram_filenames
        df['label'] = [1 if filename.replace('.png', '').split('_')[-1] == category else 0 for filename in spectrogram_filenames]
        df_index = list(range(len(df)))
        index_train_val, index_test = train_test_split(df_index, test_size = test_size, random_state = 1)
        index_train, index_val = train_test_split(index_train_val, test_size = val_size / (1 - test_size), random_state = 1)
        df['split'] = [x[1] for x in sorted(list(zip(index_train + index_val + index_test, 
                                                       ['train'] * len(index_train) + ['val'] * len(index_val) + ['test'] * len(index_test))), 
                                              key = lambda x:x[0])
                        ]
        
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
    

    
    def data_split_multiclass(categories, spectrogram_dir, train_size = 0.7, val_size = 0.15, test_size = 0.15):
        
        if sum([train_size, val_size, test_size]) != 1:
            raise ValueError("The total percentage of training/validation/testing datasets need to be 100%.")
        
        #spectrogram_filenames = [filename for filename in glob.glob(spectrogram_dir + '*.png') if filename.replace('.png', '').split('_')[-1] in categories]
        spectrogram_filenames = glob.glob(spectrogram_dir + '*.png')
        
        df = pd.DataFrame()
        df['filename'] = spectrogram_filenames
        df['label'] = [filename.replace('.png', '').split('_')[-1] if filename.replace('.png', '').split('_')[-1] in categories else '-9999' for filename in spectrogram_filenames]
        df_index = list(range(len(df)))
        index_train_val, index_test = train_test_split(df_index, test_size = test_size, random_state = 1)
        index_train, index_val = train_test_split(index_train_val, test_size = val_size / (1 - test_size), random_state = 1)
        df['split'] = [x[1] for x in sorted(list(zip(index_train + index_val + index_test, 
                                                       ['train'] * len(index_train) + ['val'] * len(index_val) + ['test'] * len(index_test))), 
                                              key = lambda x:x[0])
                        ]
        
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
    
    
    def data_split_allclasses(spectrogram_dir, train_size = 0.7, val_size = 0.15, test_size = 0.15):
        
        if sum([train_size, val_size, test_size]) != 1:
            raise ValueError("The total percentage of training/validation/testing datasets need to be 100%.")
        
        spectrogram_filenames = glob.glob(spectrogram_dir + '*.png')
        
        df = pd.DataFrame()
        df['filename'] = spectrogram_filenames
        df['label'] = [filename.replace('.png', '').split('_')[-1] for filename in spectrogram_filenames]
        df_index = list(range(len(df)))
        index_train_val, index_test = train_test_split(df_index, test_size = test_size, random_state = 1)
        index_train, index_val = train_test_split(index_train_val, test_size = val_size / (1 - test_size), random_state = 1)
        df['split'] = [x[1] for x in sorted(list(zip(index_train + index_val + index_test, 
                                                       ['train'] * len(index_train) + ['val'] * len(index_val) + ['test'] * len(index_test))), 
                                              key = lambda x:x[0])
                        ]
        
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
                    
