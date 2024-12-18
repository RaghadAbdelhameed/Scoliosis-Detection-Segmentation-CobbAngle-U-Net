import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Placeholder directories
image_dir = "/kaggle/input/spine-segmentation/Processed_Scol"
label_dir = "/kaggle/input/spine-segmentation/Processed_Scol_Line"

# Image preprocessing function (padding to make aspect ratio 1:1 and resizing)
def preprocess_image(image_path, label_path, target_size=(256, 256)):
    # Load image and label
    image = load_img(image_path, color_mode='grayscale')
    label = load_img(label_path, color_mode='grayscale')
    image = img_to_array(image)
    label = img_to_array(label)

    # Get original dimensions
    original_height, original_width = image.shape[:2]

    # Calculate padding dimensions
    max_dim = max(original_height, original_width)
    pad_height = (max_dim - original_height) // 2
    pad_width = (max_dim - original_width) // 2

    # Apply padding
    image = np.pad(
        image,
        ((pad_height, max_dim - original_height - pad_height),
         (pad_width, max_dim - original_width - pad_width),
         (0, 0)),
        mode='constant',
        constant_values=0
    )
    label = np.pad(
        label,
        ((pad_height, max_dim - original_height - pad_height),
         (pad_width, max_dim - original_width - pad_width),
         (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Resize to target size
    image = tf.image.resize(image, target_size) / 255.0
    label = tf.image.resize(label, target_size) / 255.0

    # Convert label to binary mask
    label = tf.cast(label > 0.5, tf.float32)

    return image, label

# Load data
def load_data(image_dir, label_dir, target_size=(256, 256)):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
    images, labels = [], []
    for img_path, lbl_path in zip(image_paths, label_paths):
        image, label = preprocess_image(img_path, lbl_path, target_size)
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

# U-Net model definition
def unet_model(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)

    # Encoding path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoding path
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and split data
target_size = (256, 256)
images, labels = load_data(image_dir, label_dir, target_size)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the model
model = unet_model(input_size=(target_size[0], target_size[1], 1))

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=[early_stopping]
)

# Save the model
model.save("spinal_cord_unet(with padding v2).h5")

# Plot accuracy and loss curves
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
