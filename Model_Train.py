import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

# If running on GPU, set deterministic operations to avoid non-deterministic behavior
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Model parameters
img_width = 25
img_height = 50  # ideally want to have full image res, but computer not strong enough for that
batch_size = 50

# Build the model
model = models.Sequential()
model.add(layers.Conv2D(50, (3, 3), activation='relu', input_shape=(25, 50, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(100, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(100, (3, 3), activation='relu'))

# Add Dropout layer to reduce overfitting
model.add(layers.Dropout(0.3))  # Dropout rate of 30%

model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))

# Softmax activation for multi-class classification
model.add(layers.Dense(2, activation='softmax'))  # Softmax for two classes

# Load the training dataset
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    './Train',
    labels='inferred',
    label_mode="int",  # categorical, binary
    class_names=['Normal', 'Scol'],
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_width, img_height),  # reshape if not in this size
    shuffle=True,
    seed=123,  # makes split same every time
    validation_split=0.1,
    subset='training')

# Load the validation dataset
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    './Train',
    labels='inferred',
    label_mode="int",  # categorical, binary
    class_names=['Normal', 'Scol'],
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_width, img_height),  # reshape if not in this size
    shuffle=True,
    seed=123,  # ensures the same validation split every time
    validation_split=0.1,
    subset='validation')

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # no logits needed with softmax
              metrics=['accuracy'])

# EarlyStopping callback to stop training when validation loss doesn't improve
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(ds_train, 
                    epochs=20, 
                    verbose=2, 
                    validation_data=ds_validation, 
                    callbacks=[early_stopping])  # Apply early stopping

# Save the model in .h5 format
model.save('spine_rec_model_with_dropout_earlystop.h5')

# Plot Accuracy and Loss Graphs
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
