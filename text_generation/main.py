import tensorflow as tf

import numpy as np
import os
import time



# Download the Shakespeare dataset
# Change the following line to run this code on your own data.
path_to_file = tf.keras.utils.get_file('data/shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read the data
# First, look in the text:

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters\n')

# Take a look at the first 250 characters in text
print(f'First characters in the text: {text[:250]}\n')

# The unique characters in the file
vocab = sorted(set(text))
print(f'The file has {len(vocab)} unique characters')


# Process the text
# Vectorize the text
# Before training, you need to convert the strings to a numerical representation.

# The tf.keras.layers.StringLookup layer can convert each character into a numeric ID. It just needs the text to be split into tokens first.

example_texts = ['abcdefg', 'xyz']

chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
print(f'Example of characters converted into numbers (tokenization): {example_texts} -> {chars}')

Now create the tf.keras.layers.StringLookup layer:


ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
     
It converts from tokens to character IDs:


ids = ids_from_chars(chars)
ids
     
Since the goal of this tutorial is to generate text, it will also be important to invert this representation and recover human-readable strings from it. For this you can use tf.keras.layers.StringLookup(..., invert=True).

Note: Here instead of passing the original vocabulary generated with sorted(set(text)) use the get_vocabulary() method of the tf.keras.layers.StringLookup layer so that the [UNK] tokens is set the same way.


chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
     
This layer recovers the characters from the vectors of IDs, and returns them as a tf.RaggedTensor of characters:


chars = chars_from_ids(ids)
chars
     
You can tf.strings.reduce_join to join the characters back into strings.


tf.strings.reduce_join(chars, axis=-1).numpy()
     

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
     



