import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import librosa  # for audio processing
from spafe.features.gfcc import gfcc
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Update paths as needed
DATASET_ROOT = '/Users/mruthunjai_govindaraju/Downloads/Data/malayalam/Female'
AUDIO_SUBFOLDER = "Voices"
NOISE_SUBFOLDER = "Noise"
DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)
INDEX_TSV_PATH = os.path.join(DATASET_ROOT, 'index.tsv')

# Constants
VALID_SPLIT = 0.1
SHUFFLE_SEED = 43
SAMPLING_RATE = 16000
SCALE = 0.5
BATCH_SIZE = 128
EPOCHS = 100
NUM_MFCC = 13  # Number of MFCCs to extract

# Load transcripts
index_df = pd.read_csv(INDEX_TSV_PATH, sep='\t', header=None)
index_df.columns = ['filename', 'transcript']

# SpecAugment Function
def spec_augment(spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
    spec = tf.identity(spec)
    for i in range(num_mask):
        # Frequency masking
        freq_mask = tf.cast(tf.random.uniform([], 0, freq_masking_max_percentage) * tf.shape(spec)[1], tf.int32)
        f0 = tf.cast(tf.random.uniform([], 0, tf.shape(spec)[1] - freq_mask), tf.int32)
        spec[:, f0:f0+freq_mask] = 0

        # Time masking
        time_mask = tf.cast(tf.random.uniform([], 0, time_masking_max_percentage) * tf.shape(spec)[0], tf.int32)
        t0 = tf.cast(tf.random.uniform([], 0, tf.shape(spec)[0] - time_mask), tf.int32)
        spec[t0:t0+time_mask, :] = 0

    return spec

# Augmentation with audiomentations
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

# Function to load and preprocess audio files
def load_and_preprocess_audio(file_path, feature_type='log_mel', augment_data=False):
    audio, _ = librosa.load(file_path, sr=SAMPLING_RATE)
    if augment_data:
        audio = augment(samples=audio, sample_rate=SAMPLING_RATE)
    
    if feature_type == 'log_mel':
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE, n_mels=40)
        log_mel_spec = librosa.power_to_db(mel_spec)
        return log_mel_spec.T
    
    elif feature_type == 'mfcc':
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLING_RATE, n_mfcc=NUM_MFCC)
        return mfcc.T

    elif feature_type == 'gfcc':
        gfcc_features = gfcc(sig=audio, fs=sr, num_ceps=NUM_MFCC)
        return gfcc_features

# Function to add noise
def add_noise(audio):
    noise_sample = tf.random.uniform((tf.shape(audio)[0],)) * 0.1
    noisy_audio = audio + noise_sample
    return noisy_audio

# Apply SpecAugment and Noise to the training dataset
def apply_augmentations(features, labels):
    features_augmented = spec_augment(features)
    features_noisy = add_noise(features_augmented)
    return features_noisy, labels

# Function to convert paths to dataset
def paths_to_dataset(audio_paths, labels, feature_type='log_mel'):
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: tf.numpy_function(load_and_preprocess_audio, [x, feature_type], tf.float32), num_parallel_calls=tf.data.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))

# Creating datasets
audio_paths = [os.path.join(DATASET_AUDIO_PATH, fname) for fname in index_df['filename']]
labels = index_df['transcript']
rng = np.random.RandomState(SHUFFLE_SEED)
shuffled_indices = rng.permutation(len(audio_paths))
train_indices = shuffled_indices[:-int(VALID_SPLIT * len(audio_paths))]
valid_indices = shuffled_indices[-int(VALID_SPLIT * len(audio_paths)):]

train_audio_paths = [audio_paths[i] for i in train_indices]
train_labels = [labels[i] for i in train_indices]
valid_audio_paths = [audio_paths[i] for i in valid_indices]
valid_labels = [labels[i] for i in valid_indices]

train_ds = paths_to_dataset(train_audio_paths, train_labels, feature_type='log_mel').map(apply_augmentations).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
valid_ds = paths_to_dataset(valid_audio_paths, valid_labels, feature_type='log_mel').batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Transformer Encoder Block
class ConformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, conv_kernel_size=32, rate=0.1, kernel_regularizer=None):
        super(ConformerBlock, self).__init__()
        self.ffn1 = tf.keras.layers.Dense(ff_dim, activation='relu', kernel_regularizer=kernel_regularizer)
        self.ffn2 = tf.keras.layers.Dense(embed_dim, kernel_regularizer=kernel_regularizer)

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.conv = tf.keras.layers.Conv1D(filters=2*embed_dim, kernel_size=conv_kernel_size, padding='same', activation='relu', kernel_regularizer=kernel_regularizer)
        self.conv1 = tf.keras.layers.Conv1D(filters=embed_dim, kernel_size=1, activation='relu', kernel_regularizer=kernel_regularizer)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, inputs, training=False):
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

# Building the Conformer Model with Regularization
def build_conformer_model(hp, input_shape=(None, 40), num_classes=None):
    num_conformer_blocks = hp.get('num_conformer_blocks', 4)
    embed_dim = 128
    num_heads = 4
    ff_dim = 128
    conv_kernel_size = 32
    kernel_regularizer = tf.keras.regularizers.l2(0.001)

    inputs = keras.layers.Input(shape=input_shape)
    x = inputs

    for _ in range(num_conformer_blocks):
        x = ConformerBlock(embed_dim, num_heads, ff_dim, conv_kernel_size, kernel_regularizer=kernel_regularizer)(x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=kernel_regularizer)(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=kernel_regularizer)(x)
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=kernel_regularizer)(x)

    model = keras.models.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.get(hp.get('optimizer', 'adam')),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

num_classes = len(np.unique(labels))
# Define hyperparameter grid
hyperparameter_grid = {
    'learning_rate': [0.001, 0.0001],
    'optimizer': ['adam', 'rmsprop'],
    'num_conformer_blocks': [3, 4, 5],
}

# Create a Keras Tuner object
tuner = keras.tuners.BayesianOptimization(
    build_conformer_model,
    objective='val_accuracy',
    max_trials=10,
    hyperparameters=hyperparameter_grid,
)

# Early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

# Train the model with hyperparameter tuning
tuner.search(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr],
)

# Get the best hyperparameter configuration
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build and train the model with the best hyperparameters
model = build_conformer_model(input_shape, num_classes, **best_hyperparameters)

model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS)

# Evaluate the model with the best hyperparameters
model.evaluate(valid_ds)

model = build_conformer_model((None, 40), num_classes)  # Adjust input shape based on feature extraction

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training the model
history = model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds)

# Evaluate the model
print(model.evaluate(valid_ds))
