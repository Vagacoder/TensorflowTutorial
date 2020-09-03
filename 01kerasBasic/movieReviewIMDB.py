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
# ! Step 1. Load IMDB dataset
# * Load IMDB dataset
# ? parameter num_words=10000 keeps 10000 words of highest frequency.
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = \
    imdb.load_data(num_words=10000)

print('Loading done')

# %%
# * Explore Dataset
# ? Training entry number: 25000
print('Training entries: {}, labels: {}'.format(len(train_data), \
    len(train_labels)))

# ? review is transformed to integers, each int represents a word 
print(train_data[0])

print(len(train_data[0]))
print(len(train_data[1]))

#%%
# ! Step 2. Setup word index and reverse index
# * Get word index and reverse index
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
# * Inspect data
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
# ! Step 3. Prepare dataset
# * Prepare dataset
# ? Review is a list of integers now. We need transform it to scalar
# ? Two ways to do it:
# ? 1. transform list to one-hot coding, since max length of review is 10000 (see
# ? imdb.load_data() ), we have to fill a list of 10000 for each review, it need 
# ? huge memory
# ? 2. we fill the list of each review to ensure they are in same length.

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                    value=word_index['<PAD>'], padding='post',maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                    value=word_index['<PAD>'], padding='post', maxlen=256)

print(len(train_data[0]))
print(len(train_data[1]))

print(train_data[0])

# %%
# ! Step 4. Setup Neural Network, add layers
# * Modeling
# * How to setup neural network? 2 major factors we need consider
# ? 1. how many layers
# ? 2. how many hidden units in each layer?
# * Input is a list of word index, output is a tag 0 or 1.

# ? size of each list (each review max length is 10000)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# * Summary of layers
# ? 1. First layer is Embedding layer. 
# ? 2. Second layer is Global Average Pooling 1D layer
# ? 3. Third layer is Dense layer with 16 units, using relu function
# ? 4. Forth layer is Dense layer with 1 output units, using sigmoid function

# %%
# ! Step 5. Compile Neural Network, add optimizerm, loss function
# * Compile model with loss function and optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
# ! Step 6. Split training dataset to partial training dataset and validate dataset
# * Create a validate dataset
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# %%
# ! Step 7. Train Neural Network
# * training model
history = model.fit(partial_x_train, partial_y_train,
                    epochs=40, batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# %%
# ! Step 8. Evaluate model
# * Evaluate model
results = model.evaluate(test_data, test_labels, verbose=2)

print(results)

#%%
# ! Step 9. (optional) Plot the history
# * Create a plot of loss vs. epoch
history_dict = history.history
history_dict.keys()
# %%
import matplotlib.pylab as plt 

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) +1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Taining and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# %%
# * Create a plot of accuracy vs. epoch

plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# %%
