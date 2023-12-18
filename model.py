import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import librosa
from spafe.features.gfcc import gfcc
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

class AudioDataProcessor:
    """
    Audio Data Processor for loading and preprocessing audio data.

    Attributes:
        dataset_root (str): Root directory of the dataset.
        audio_subfolder (str): Subfolder containing audio files.
        noise_subfolder (str): Subfolder containing noise files.
        index_tsv_path (str): Path to the index TSV file.
        sampling_rate (int): Sampling rate for audio processing.
        num_mfcc (int): Number of Mel-frequency cepstral coefficients.

    Methods:
        load_index: Loads the index file and returns a DataFrame.
        split_dataset: Splits the dataset into training and validation sets.
    """
    def __init__(self, dataset_root, audio_subfolder, noise_subfolder, index_tsv_path, sampling_rate=16000, num_mfcc=13):
        self.dataset_root = dataset_root
        self.audio_subfolder = audio_subfolder
        self.noise_subfolder = noise_subfolder
        self.index_tsv_path = index_tsv_path
        self.sampling_rate = sampling_rate
        self.num_mfcc = num_mfcc

    def load_index(self):
        """Loads the index TSV file."""
        index_df = pd.read_csv(self.index_tsv_path, sep='\t', header=None)
        index_df.columns = ['filename', 'transcript']
        return index_df

    def split_dataset(self, valid_split=0.1, shuffle_seed=43):
        """Splits the dataset into training and validation sets."""
        index_df = self.load_index()
        audio_paths = [os.path.join(self.dataset_root, self.audio_subfolder, fname) for fname in index_df['filename']]
        labels = index_df['transcript']
        rng = np.random.RandomState(shuffle_seed)
        shuffled_indices = rng.permutation(len(audio_paths))
        train_indices = shuffled_indices[:-int(valid_split * len(audio_paths))]
        valid_indices = shuffled_indices[-int(valid_split * len(audio_paths)):]

        train_audio_paths = [audio_paths[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        valid_audio_paths = [audio_paths[i] for i in valid_indices]
        valid_labels = [labels[i] for i in valid_indices]

        return train_audio_paths, train_labels, valid_audio_paths, valid_labels


class ConformerBlock(keras.layers.Layer):
    """
    Conformer block for the Conformer model.

    Attributes:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of heads in the multi-head attention.
        ff_dim (int): Dimension of feed-forward network.
        conv_kernel_size (int): Kernel size for the convolution layer.
        rate (float): Dropout rate.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.

    Methods:
        call: Performs the computation from inputs to outputs.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, conv_kernel_size=32, rate=0.1, kernel_regularizer=None):
        super(ConformerBlock, self).__init__()
        self.ffn1 = Dense(ff_dim, activation='relu', kernel_regularizer=kernel_regularizer)
        self.ffn2 = Dense(embed_dim, kernel_regularizer=kernel_regularizer)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.conv = keras.layers.Conv1D(filters=2*embed_dim, kernel_size=conv_kernel_size, padding='same', activation='relu', kernel_regularizer=kernel_regularizer)
        self.conv1 = keras.layers.Conv1D(filters=embed_dim, kernel_size=1, activation='relu', kernel_regularizer=kernel_regularizer)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, inputs, training=False):
        """Performs the computation of the Conformer block."""
        ffn_output = self.ffn1(inputs)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout1(ffn_output, training=training)
        out1 = self.layernorm1(inputs + ffn_output)

        attn_output = self.att(out1, out1)
        attn_output = self.dropout2(attn_output, training=training)
        out2 = self.layernorm2(out1 + attn_output)

        conv_output = self.conv(out2)
        conv_output = self.conv1(conv_output)
        conv_output = self.dropout3(conv_output, training=training)
        out3 = self.layernorm3(out2 + conv_output)

        return out3


class ConformerModel:
    """
    Conformer Model for speech recognition.

    Attributes:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of output classes.

    Methods:
        build_model: Builds the Conformer model.
    """
    def __init__(self, input_shape=(None, 40), num_classes=None):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self, hp):
        """Builds the Conformer model with hyperparameters."""
        num_conformer_blocks = hp.get('num_conformer_blocks', 4)
        embed_dim = 128
        num_heads = 4
        ff_dim = 128
        conv_kernel_size = 32
        kernel_regularizer = keras.regularizers.l2(0.001)

        inputs = keras.layers.Input(shape=self.input_shape)
        x = inputs
        for _ in range(num_conformer_blocks):
            x = ConformerBlock(embed_dim, num_heads, ff_dim, conv_kernel_size, kernel_regularizer=kernel_regularizer)(x)

        x = keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=kernel_regularizer)(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu', kernel_regularizer=kernel_regularizer)(x)
        outputs = Dense(self.num_classes, activation='softmax', kernel_regularizer=kernel_regularizer)(x)

        model = keras.models.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.get(hp.get('optimizer', 'adam')),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model

class AudioAugmenter:
    """
    Audio Augmenter for applying various augmentations to audio data.

    Methods:
        spec_augment: Applies SpecAugment to the given spectrogram.
        add_noise: Adds noise to the audio.
        augment: Applies a series of augmentations using audiomentations.
    """
    @staticmethod
    def spec_augment(spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
        """Applies SpecAugment to the given spectrogram."""
        spec = tf.identity(spec)
        for i in range(num_mask):
            freq_mask = tf.cast(tf.random.uniform([], 0, freq_masking_max_percentage) * tf.shape(spec)[1], tf.int32)
            f0 = tf.cast(tf.random.uniform([], 0, tf.shape(spec)[1] - freq_mask), tf.int32)
            spec[:, f0:f0+freq_mask] = 0

            time_mask = tf.cast(tf.random.uniform([], 0, time_masking_max_percentage) * tf.shape(spec)[0], tf.int32)
            t0 = tf.cast(tf.random.uniform([], 0, tf.shape(spec)[0] - time_mask), tf.int32)
            spec[t0:t0+time_mask, :] = 0

        return spec

    @staticmethod
    def add_noise(audio):
        """Adds noise to the audio."""
        noise_sample = tf.random.uniform((tf.shape(audio)[0],)) * 0.1
        noisy_audio = audio + noise_sample
        return noisy_audio

    @staticmethod
    def augment():
        """Applies a series of augmentations using audiomentations."""
        return Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        ])


class AudioFeatureExtractor:
    """
    Audio Feature Extractor for extracting features from audio data.

    Methods:
        load_and_preprocess_audio: Loads and preprocesses audio files.
        apply_augmentations: Applies augmentations to features and labels.
        paths_to_dataset: Converts paths to a TensorFlow dataset.
    """
    @staticmethod
    def load_and_preprocess_audio(file_path, feature_type='log_mel', augment_data=False, sampling_rate=16000):
        """Loads and preprocesses audio files."""
        audio, _ = librosa.load(file_path, sr=sampling_rate)
        if augment_data:
            audio = AudioAugmenter.augment()(samples=audio, sample_rate=sampling_rate)

        if feature_type == 'log_mel':
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=40)
            log_mel_spec = librosa.power_to_db(mel_spec)
            return log_mel_spec.T

        elif feature_type == 'mfcc':
            mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=13)
            return mfcc.T

        elif feature_type == 'gfcc':
            gfcc_features = gfcc(sig=audio, fs=sampling_rate, num_ceps=13)
            return gfcc_features

    @staticmethod
    def apply_augmentations(features, labels):
        """Applies augmentations to features and labels."""
        features_augmented = AudioAugmenter.spec_augment(features)
        features_noisy = AudioAugmenter.add_noise(features_augmented)
        return features_noisy, labels

    @staticmethod
    def paths_to_dataset(audio_paths, labels, feature_type='log_mel', sampling_rate=16000):
        """Converts paths to a TensorFlow dataset."""
        path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
        audio_ds = path_ds.map(lambda x: tf.numpy_function(
            lambda file_path: AudioFeatureExtractor.load_and_preprocess_audio(
                file_path, feature_type, False, sampling_rate
            ), [x], tf.float32
        ), num_parallel_calls=tf.data.AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        return tf.data.Dataset.zip((audio_ds, label_ds))


class HyperparameterTuner:
    """
    Hyperparameter Tuner using Keras Tuner.

    Attributes:
        build_model_fn: Function to build the model.
        objective (str): Objective for the tuner to optimize.
        max_trials (int): Maximum number of trials for hyperparameter tuning.
        hyperparameter_grid (dict): Hyperparameter grid for tuning.

    Methods:
        create_tuner: Creates a tuner instance.
    """
    def __init__(self, build_model_fn, objective, max_trials, hyperparameter_grid):
        self.build_model_fn = build_model_fn
        self.objective = objective
        self.max_trials = max_trials
        self.hyperparameter_grid = hyperparameter_grid

    def create_tuner(self):
        """Creates a tuner instance."""
        return keras.tuners.BayesianOptimization(
            self.build_model_fn,
            objective=self.objective,
            max_trials=self.max_trials,
            hyperparameters=self.hyperparameter_grid,
        )


class ConformerTrainer:
    """
    Trainer for Conformer Model.

    Attributes:
        model (keras.Model): The Conformer model to be trained.
        train_data (tf.data.Dataset): Training data.
        valid_data (tf.data.Dataset): Validation data.
        epochs (int): Number of epochs to train.

    Methods:
        train_model: Trains the model.
        evaluate_model: Evaluates the model.
    """
    def __init__(self, model, train_data, valid_data, epochs):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.epochs = epochs

    def train_model(self):
        """Trains the Conformer model."""
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
        self.model.fit(self.train_data, validation_data=self.valid_data, epochs=self.epochs, callbacks=[early_stopping, reduce_lr])

    def evaluate_model(self):
        """Evaluates the Conformer model."""
        return self.model.evaluate(self.valid_data)

