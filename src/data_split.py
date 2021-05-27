#!/usr/bin/env python3
""" spectrogram.py: Utilities for data pre-processing of binary classification
"""

from sklearn.model_selection import train_test_split


class Data_Split:
    """ 
    ## TODO: shuffle and stratify
    """ 
    def data_split_binary(target_id, specctrogram_vector, spectrogram_filenames, train_size = 0.7, val_size = 0.15, test_size = 0.15):
        
        if sum([train_size, val_size, test_size]) != 1:
            raise ValueError("The total percentage of training/validation/testing datasets need to be 100%.")
        
        
        binary_labels = [1 if filename.split('.')[0].split('_')[-1] == target_id else 0 for filename in spectrogram_filenames]
        x_train_val, x_test, y_train_val, y_test, spectrogram_filenames_train_val, spectrogram_filenames_test = train_test_split(specctrogram_vector,
                                                                                                                                 binary_labels, 
                                                                                                                                 spectrogram_filenames, 
                                                                                                                                 test_size = test_size, 
                                                                                                                                 random_state = 1)
        
        x_train, x_val, y_train, y_val, spectrogram_filenames_train, spectrogram_filenames_val = train_test_split(x_train_val,
                                                                                                                  y_train_val,
                                                                                                                  spectrogram_filenames_train_val,
                                                                                                                  test_size = val_size / (1 - test_size), 
                                                                                                                  random_state = 1)                    
        return x_train, x_val, x_test, y_train, y_val, y_test, spectrogram_filenames_train, spectrogram_filenames_val, spectrogram_filenames_test
                    
