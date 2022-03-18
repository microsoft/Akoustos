'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

from cProfile import label
import os

os.environ["CUDA_VISIBLE_DEVICES"]="7"

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

import warnings
warnings.filterwarnings("ignore")

from load_data import Load_Data
from data_visualization import Data_Visualization
from sound_event_detection import Sound_Event_Detection
from spectrogram import Spectrogram
from data_split import Data_Split

from model.binary_classification import Binary_Classification_Models
from model.multiclass_classification import Multiclass_Classification_Models
from model.binary_classification import Binary_Classification_Scoring
from model.multiclass_classification import Multiclass_Classification_Scoring

class Akoustos:
    def __init__(self, raw_audio_directory, labeled_data_directory, spectrogram_output_dir, models_dir):
        """Creates a new instance of the Akoustos wrapper, you need to provide paths for where your data is.

        Args:
            raw_audio_directory: Directory containing raw audio files. Supported formats: flac, wav.

            labeled_data_directory: Directory containing labeled data files. Supported formats: xlsx, csv, txt.
            
            spectrogram_output_dir: Directory where you wish to save generated spectrograms.

            models_dir: Directory where you wish to save generated models.

        """
        self.spectrogram_dir =  spectrogram_output_dir
        if not os.path.exists(self.spectrogram_dir):
            os.makedirs(self.spectrogram_dir)

        self.models_dir = models_dir
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)    
            
        self.audio_dir = raw_audio_directory
        self.labeled_data_dir = labeled_data_directory

    
    #################
    ### Data load ###
    #################
    
    def load_audio_data_to_score(self, raw_audio_directory=None):
        """Loads new audio data and returns a list with the imported files locations.

        Args:
            raw_audio_directory: Directory containing raw audio files. Supported formats: flac, wav.

        """
        if raw_audio_directory is None:
            raw_audio_directory = self.audio_dir
        return Load_Data.audio_filenames(raw_audio_directory)    

    def load_labeled_data(self, labeled_data_directory=None):
        """Loads new labeled data and returns a list with the imported files locations.

        Args:
            labeled_data_directory: Directory containing labeled data files. Supported formats: xlsx, csv, txt.

        """
        if labeled_data_directory is None:
            labeled_data_directory = self.labeled_data_dir
        return Load_Data.labeled_data(labeled_data_directory)    

    ##########################
    ### Data visualization ###
    ##########################

    def view_histogram_call_duration(self):
        """Generate and show a graphic representing the histogram for labeled calls.

        """
        Data_Visualization.histogram_call_duration(labeled_data_dir = self.labeled_data_dir, audio_dir = self.audio_dir)

    def generate_sample_spectrograms(self, length_in_seconds, color_maps = 'viridis'):
        """Generate sample spectrograms from the loaded data files.

        Args:
            length_in_seconds: Length in seconds of the spectrograms to generate. This is how long of the audio clip we want to analyze.

            color_maps: Color map for the output spectrogram. We have set a default value of 'viridis'; if you wish, you can chose different options from the available color maps, see https://matplotlib.org/stable/tutorials/colors/colormaps.html 

        """
        Data_Visualization.sample_spectrograms(labeled_data_dir = self.labeled_data_dir, audio_dir = self.audio_dir, length_in_seconds = length_in_seconds, cmap = color_maps)

    ###########################
    ### Data pre-processing ###
    ###########################

    def sound_event_detection_for_single_audio_file(self, file_name):
        """Detect a non empty sound event in a given file.

        Args:
            file_name: File name to run audio detection on.

        """

        self.sound_event_detection = Sound_Event_Detection(labeled_data_filenames=self.load_labeled_data(self.labeled_data_dir), audio_filenames=self.load_audio_data_to_score(self.audio_dir))
        return self.sound_event_detection.sound_event_detection_for_single_audio_file(file_name)

    def sound_event_detection(self):
        """Detect a non empty sound event in multiple given files. By default this code will run against the data located in the raw_audio_directory.
        Args:
        """
        sound_event_detection = Sound_Event_Detection(labeled_data_filenames=self.load_labeled_data(self.labeled_data_dir), audio_filenames=self.load_audio_data_to_score(self.audio_dir))
        annotated_data = sound_event_detection.sound_event_detection_for_all_audio_files()
        return annotated_data

    def generate_spectrograms(self, spectrogram_duration_in_seconds, clear_spectrograms_space = True, axis = False, sr = 22050, hop_length = 512, fmin = None, x_axis = 'time', y_axis = 'linear', cmap = 'viridis'):
        """Generate spectrograms and save them to the spectrogram_dir, under: your_base_dir/Extracted_Spectrogram. 

        Returns a dataframe containing the generated spectrogram information.

        Args:
            spectrogram_duration_in_seconds: 
            clear_spectrograms_space: Flag indicating whether to clear previously generated spectrograms in the spectrograms directory.
            axis:
            sr:
            hop_length:
            fmin:
            x_axis:
            y_axis:
            cmap: Selected color map for this spectrograms.
            labeled_data: You can pass a Pandas Dataframe if you have it, you can generate it by running the sound_event_detection() function from this class.

        """
        
        labeled_data_filenames = self.load_labeled_data(self.labeled_data_dir)   

        # If we have the labeled data in a df, just generate the spectrograms off that data. No need to re-run sound_event_detection.

        # if len(audio_filenames) == 0 or audio_filenames is None:
        #     print("Can't generate spectrograms. No audio files detected in {0}".format(audio_filenames))
        #     return
        
        # if len(labeled_data_filenames) == 0 or labeled_data_filenames is None:
        #     print("Can't generate spectrograms. No labeled files detected in {0}".format(labeled_data_filenames))
        #     return
        
        spectrogram = Spectrogram(raw_audio_dir = self.audio_dir, 
                                    spectrogram_duration = spectrogram_duration_in_seconds, 
                                    labeled_data = labeled_data_filenames, 
                                    save_to_dir = self.spectrogram_dir,
                                    axis = axis, 
                                    sr = sr, 
                                    hop_length = hop_length, 
                                    fmin = fmin, 
                                    x_axis = x_axis, 
                                    y_axis = y_axis, 
                                    cmap = cmap) 

        if clear_spectrograms_space is True:
            spectrogram.clear_space(self.spectrogram_dir)

        spectrogram.generate_spectrograms_parallel()
        
    ################
    ### Modeling ###
    ################

    def train_binary_classification_model(self, categories, model_name = 'Customized_CNN', model_version = 'v1', batch_size = 32, optimizer = 'Adam', learning_rate = 0.008, lr_decay = False, num_epochs = 25, train_size = 0.7, val_size = 0.15, test_size = 0.15, by = 'random', include_no_label_category = True):
        """Train model using binary classification.

        Generates the model and saves it to the selected models directory.

        Args:
            categories: Array conaining the categories you wish to use.
            model_name: Possible values are: Customized_CNN, Resnet18, Resnet34, Resnet50, Resnet101, Resnet152, Alexnet, VGG11, VGG13, VGG16, VGG19, Densenet121, Densenet169, Densenet201, Squeezenet1_0.
            model_version: 
            batch_size: 
            optimizer: 
            learning_rate: 
            lr_decay: 
            num_epochs: 
            train_size: 
            val_size: 
            test_size: 
            by: 
            include_no_label_category: 

        """
        data = Data_Split.data_split(categories, self.spectrogram_dir, train_size = train_size, val_size = val_size, test_size = test_size, by = by, include_no_label_category = include_no_label_category)
        Binary_Classification_Models.train_model(data = data, 
                                                 model_name = model_name, 
                                                 model_version = model_version,
                                                 model_dir = self.models_dir,
                                                 batch_size = batch_size,  
                                                 optimizer = optimizer, 
                                                 learning_rate = learning_rate, 
                                                 lr_decay = lr_decay, 
                                                 num_epochs = num_epochs)

    def train_multiclass_classification_model(self, categories, model_name = 'Customized_CNN', model_version = 'v1', batch_size = 32, pretrained = True, optimizer = 'Adam', learning_rate = 0.008, lr_decay = False, num_epochs = 25, train_size = 0.7, val_size = 0.15, test_size = 0.15, by = 'random', include_no_label_category = True):
        """Train model using multi-class classification.

        Returns a dataframe containing the generated spectrogram information.

        Args:
            categories: Array conaining the categories you wish to use. Pass [] if you wish to classify all categories in the data.

        """
        data = Data_Split.data_split(categories, self.spectrogram_dir, train_size = train_size, val_size = val_size, test_size = test_size, by = by, include_no_label_category = include_no_label_category)
        Multiclass_Classification_Models.train_model(data = data, 
                                                        model_name = model_name, 
                                                        model_version = model_version,
                                                        model_dir = self.models_dir,
                                                        batch_size = batch_size,  
                                                        pretrained = pretrained, 
                                                        optimizer = optimizer, 
                                                        learning_rate = learning_rate, 
                                                        lr_decay = lr_decay, 
                                                        num_epochs = num_epochs)

    ###############
    ### Scoring ###
    ###############

    def score_binary_classification_dataset(self, spectrogram_directory_to_score, saved_model_to_use):
        """Score binary classification model.

        Returns a dataframe containing the generated spectrogram information.

        Args:
            spectrogram_directory_to_score: Path to where the spectrograms we want to score are located.
            saved_model_to_use: Model to use for scoring phase. This is the model you generated and saved in the train step.

        """
        binary_classification_scoring = Binary_Classification_Scoring()
        return binary_classification_scoring.score_new_dataset(spectrogram_dir_to_score = spectrogram_directory_to_score, 
                                                                    model_dir = self.models_dir, 
                                                                    saved_model = saved_model_to_use)

    def score_multiclass_classification_dataset(self, spectrogram_directory_to_score, categories, saved_model):
        """Score multi-class classification model.

        Returns a dataframe containing the generated spectrogram information.

        Args:
            spectrogram_directory_to_score: Path to where the spectrograms we want to score are located.

            categories: Array conaining the categories you wish to use. Pass [] if you wish to classify all categories in the data.

        """
        return Multiclass_Classification_Scoring.score_new_dataset(categories = categories,
                                                                    spectrogram_dir_to_score = spectrogram_directory_to_score, 
                                                                    model_dir = self.models_dir, 
                                                                    saved_model = saved_model)
