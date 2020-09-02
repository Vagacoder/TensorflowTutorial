#
# * Basic classification: Classify images of clothing

#%%
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# * Fashion MNIST is replacement of classic MNIST dataset - often used as the
# * "Hello world" of machine learning for computer vision.
# ! Step 1. get dataset
fashion_mnist = keras.datasets.fashion_mnist

# ? train_images and train_labels arrays are the training set
# ? test_images and test_labels arrays are test set
# ? images are 28 x 28 Numpy arrays, with pixel values ranging from 0 to 255.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouse', 'Pullover', 'Dress', 'Coat', 'Sandal',
    'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#%%
# * Explore data
# ? train set is numpy 3D array, it has 6000 images, each is 28 x 28 pixels
print(type(train_images))
print(train_images.shape)

# ? train label has also 6000 labels, it is numpy 1D array.
print(type(train_labels))
print(train_labels.shape)
print(len(train_labels))
print(train_labels)

print(test_images.shape)
print(len(test_labels))

# %%
# * Show sample image from train dataset
plt.figure()
plt.imshow(train_images[0])
# plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

# %%
# * Preprocess the data
# ! Step 2. Normalize dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

#%%
# * Show processed train dataset
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()

# %%
# * Build the model
# ? Setup layers
# ! Step 3. Setup neural network layers, 
# ! 3.1. Setup flatten 
# ! 3.2. Setup hidden layer and activation function
# ! 3.3. Setup Output layer, but no output function
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

#%%
# ? Compile the model
# ! Step 4. Compile neural network
# ! 4.1. Setup optimizer (how to update gradient)
# ! 4.2. Setup loss function 
# ! 4.3. Setup evaluation metrics
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

#%%
# * Train the model
# ? Feed the model
# ! Step 5. Training: feed training dataset to neural network
# ! 5.1. training dataset and labels
# ! 5.2. epoch numbers
model.fit(train_images, train_labels, epochs=10)


# %%
# ? Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc);
print('\nTest loss:', test_loss);


# %%
# ? Make predictions
# ! Step 6. After training, make prediction
# ! 6.1. Setup output function for outputlayer
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

predictions = probability_model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))

print(test_labels[0])

totalNumberOfPredictions = len(predictions)
correctPredictions = 0
for i in range(totalNumberOfPredictions):
    if(np.argmax(predictions[i]) == test_labels[i]):
        correctPredictions+=1

print('Accuracy is {:.3f}'.format(
    correctPredictions/totalNumberOfPredictions
    ))

# %%
# ? Graph functions
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.2f}% ({})".format(class_names[predicted_label],
                                        100*np.max(predictions_array),
                                        class_names[true_label],
                                        color=color
                                        ))


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


#%%
# * Verify predictions
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)

plt.tight_layout()
plt.show()


# %%
# * Use the trained model
img = test_images[1]
print(img.shape)

img = np.expand_dims(img, 0)
print(img.shape)

predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

print(np.argmax(predictions_single[0]))

