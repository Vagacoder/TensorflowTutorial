#
# * Basic Text Classification
# ! This is based on English version

#%%
# * Import libraries
import os, re, shutil, string
import matplotlib.pyplot as plt 
import tensorflow as tf 

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

print(tf.__version__)

# %%
# * Sentiment analysis
# ? This notebook trains a sentiment analysis model to classify movie reviews as 
# ? positive or negative, based on the text of the review. This is an example of 
# ? binary—or two-class—classification, an important and widely applicable kind 
# ? of machine learning problem.

# ? IMDB review dataset contains text of 50000 reviews. Split it into 25000 reviews
# ? for training and 25000 fro testing. Both are balanced, both contain equal number
# ? of positive and negative reviews

# * Download IMDB dataset
url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

# %%
# * Explore IMDB dataset
print(dataset_dir) 
print(os.listdir(dataset_dir))

train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())


# %%
# * Load the dataset
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)


# %%
