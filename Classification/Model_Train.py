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
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Model parameters
img_width = 25
img_height = 50  
batch_size = 50

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# Load the training dataset
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    './Train',
    labels='inferred',
    label_mode="binary",  # ✔ Ensures correct label format for binary classification
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_width, img_height),  
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset='training')

# Load the validation dataset
ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    './Train',
    labels='inferred',
    label_mode="binary",  # ✔ Binary labels
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_width, img_height),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset='validation')

# Apply Data Augmentation
ds_train = ds_train.map(lambda x, y: (data_augmentation(x, training=True), y))

# Build the improved CNN model
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.Dropout(0.3),  # Dropout to reduce overfitting

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # ✔ Correct output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  # ✔ Correct loss function
              metrics=['accuracy'])

# EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(ds_train, 
                    epochs=20, 
                    verbose=2, 
                    validation_data=ds_validation, 
                    callbacks=[early_stopping])

# Save the trained model
model.save('spine_rec_model_optimized.h5')

# Plot Accuracy and Loss Graphs
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))

# Plot accuracy
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
