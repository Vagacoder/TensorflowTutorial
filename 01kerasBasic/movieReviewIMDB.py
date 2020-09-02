#
# * Basic text classification 
# ! this is based on Chinese version, which is different from English version
# ? IMDB dataset includes 50000 movie reviews, 25000 for training, 25000 for test
# ? This nottebook will categories reviews as Positive and Negative, a binary categroy

#%%
# * Import library
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# %%
# * Load IMDB dataset
# ? parameter num_words=10000 keeps 10000 words of highest frequency.
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# %%
# * Explore Dataset
print('Training entries: {}, labels: {}'.format(len(train_data), len(train_labels)))

# ? review is transformed to integers, each int represents a word 
print(train_data[0])

print(len(train_data[0]))
print(len(train_data[1]))

# ? Transform int back word
# ? a dictionary from word to int
word_index = imdb.get_word_index()

# ? Note: original word_index values starting from 1
for key, val in word_index.items():
    if val == 1:
        print((key, val))
        break

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


# ? a reverse dictionary from index to word
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# ? Help function show review in text
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


decode_review(train_data[0])

# %%
# ? check contents of word_index and reverse_word_index
for key, val in word_index.items():
    if val == '?':
        print((key, val))
        break

print(type(reverse_word_index))
# print(reverse_word_index)

for key, val in reverse_word_index.items():
    if key == 4:
        print((key, val))
        break


# %%
# * Prepare dataset
