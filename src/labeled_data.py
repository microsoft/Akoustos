#!/usr/bin/env python3
""" labeled_data.py: Utilities for dealing with labeled data
"""
import pandas as pd
import numpy as np
import warnings


class Labeled_Data:
    
    def load(filenames: list):
        """ 
        TODO: handle exceptions
        """
        all_labeled_data = pd.DataFrame()
        for filename in filenames:
            if filename.endswith('xlsx'):
                labeled_data = pd.read_excel(filename)
            elif filename.endswith('csv'):
                labeled_data = pd.read_csv(filename)
            all_labeled_data = all_labeled_data.append(labeled_data, ignore_index=True)
    
        return all_labeled_data

