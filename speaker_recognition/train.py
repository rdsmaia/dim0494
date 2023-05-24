import os
import pathlib

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


DATASET_PATH = 'ptBR_multi_v01'
SAMPLING_FREQ = 24000


def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)


if __name__ == "__main__":

	# check data dir
	data_dir = pathlib.Path(DATASET_PATH)
	assert data_dir.exists(), f'{DATAPATH} not found.'

	# list speakers
	speakers = np.array(tf.io.gfile.listdir(str(data_dir)))
	print(f'\nSpeakers: {speakers}\n')
	print(f'Total of {len(speakers)} speakers\n')

	# create dataset
	train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
		directory=data_dir,
		batch_size=16,
		validation_split=0.2,
		seed=0,
		output_sequence_length=SAMPLING_FREQ,
		subset='both')

	# list classes: speaker names
	label_names = np.array(train_ds.class_names)
	print(f'\nLabel names: {label_names}\n')

	# print format of the data
	print(f'Specs of the batches: {train_ds.element_spec}\n')

	# removes one dimension from audio
	train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
	val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

	# split the validation data into test and validation
	test_ds = val_ds.shard(num_shards=2, index=0)
	val_ds = val_ds.shard(num_shards=2, index=1)

	# check the format of the audio and labels
	for example_audio, example_labels in train_ds.take(1):
		print(example_audio.shape)
		print(example_labels.shape)

	# extract spectrograms from speech
	train_spectrogram_ds = make_spec_ds(train_ds)
	val_spectrogram_ds = make_spec_ds(val_ds)
	test_spectrogram_ds = make_spec_ds(test_ds)

	# look at the spectrogram batches
	for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
		break

	# prefecth batches
	train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
	val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
	test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

	# get the input shape (to input at the network)
	input_shape = example_spectrograms.shape[1:]
	print(f'\nInput shape: {input_shape}\n')
	num_labels = len(label_names)

	# Instantiate the `tf.keras.layers.Normalization` layer.
	norm_layer = layers.Normalization()
	# Fit the state of the layer to the spectrograms
	# with `Normalization.adapt`.
	norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

	# build model
	model = models.Sequential([
		layers.Input(shape=input_shape),
		# Downsample the input.
		layers.Resizing(32, 32),
		# Normalize.
		norm_layer,
		layers.Conv2D(32, 3, activation='relu'),
		layers.Conv2D(64, 3, activation='relu'),
		layers.MaxPooling2D(),
		layers.Dropout(0.25),
		layers.Flatten(),
		layers.Dense(128, activation='relu'),
		layers.Dropout(0.5),
		layers.Dense(num_labels),
	])
	model.summary()

	# compile model
	model.compile(
		optimizer=tf.keras.optimizers.Adam(),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy'],
	)

	# train the model
	EPOCHS = 10
	history = model.fit(
		train_spectrogram_ds,
		validation_data=val_spectrogram_ds,
		epochs=EPOCHS,
		callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
		)

	# evaluate the model
	model.evaluate(test_spectrogram_ds, return_dict=True)

