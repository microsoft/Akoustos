#!/usr/bin/env python3
""" audio.py: Utilities for dealing with audio files
"""

import librosa
import librosa.display
from matplotlib import pyplot as plt
import soundfile
import numpy as np
import pandas as pd
from pathlib import Path


class Audio:
    """Container for audio samples

    Initializing an `Audio` object directly requires the specification of the
    sample rate. Use `Audio.from_file` or `Audio.from_bytesio` with
    `sample_rate=None` to use a native sampling rate.

    Args:
        samples (np.array):     The audio samples
        sample_rate (integer):  The sampling rate for the audio samples

    Returns:
        An initialized `Audio` object
    """

    def __init__(self, samples, sample_rate):
        # Do not move these lines; it will break Pytorch training
        self.samples = samples
        self.sample_rate = sample_rate

    @classmethod
    def load(cls, path):
        """Load audio from files

        Deal with the various possible input types to load an audio
        file and generate a spectrogram

        Args:
            path (str, Path): path to an audio file

        Returns:
            Audio: attributes samples and sample_rate
        """
        
        samples, sample_rate = librosa.load(path)
        return cls(samples, sample_rate)

    
    def time_to_sample(self, time):
        """Given a time, convert it to the corresponding sample
        Args:
            time: The time to multiply with the sample_rate
        Returns:
            sample: The rounded sample
        """
        return int(time * self.sample_rate)
    
    
    def trim(self, start_time, end_time):
        """Trim Audio object in time

        Args:
            start_time: time in seconds for start of extracted clip
            end_time: time in seconds for end of extracted clip
        Returns:
            a new Audio object containing samples from start_time to end_time
        """
        start_sample = self.time_to_sample(start_time)
        end_sample = self.time_to_sample(end_time)
        samples_trimmed = self.samples[start_sample:end_sample]
        return Audio(
            samples_trimmed,
            self.sample_rate
        )

    def duration(self):
        """Return duration of Audio

        Returns:
            duration (float): The duration of the Audio
        """
        return len(self.samples) / self.sample_rate


    def generate_spectrogram(self, axis=False, filename=None, melspectrogram = False, cmap = 'viridis'):
        fig, ax = plt.subplots()
        if axis == False:
            plt.axis('off')
        
        if melspectrogram == False:
            D = librosa.stft(self.samples)  # STFT of y
            S_dB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='linear', sr=self.sample_rate, ax=ax, cmap = cmap)
  
        elif melspectrogram == True:
            S = librosa.feature.melspectrogram(y=self.samples, sr=self.sample_rate, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=self.sample_rate, ax=ax, cmap = cmap)
            
        if filename:
            image_path = Path(filename)
            fig.savefig(image_path)
            plt.close()   

        

    
    