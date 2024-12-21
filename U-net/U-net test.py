import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Placeholder for model path
model_path = "D:/Reseach day/ARC/U-net/spinal_cord_unet.h5"  # Replace with the path to your saved model

# Placeholder for image and label directories
image_dir = "D:/Reseach day/ARC/Processed_Scol(test)"  # Replace with your test image directory
label_dir = "D:/Reseach day/ARC/Processed_Scol(test)Line"  # Replace with your test label directory

# Load the trained model
model = load_model(model_path)

# Image preprocessing function (padding and resizing)
def preprocess_image(image_path, label_path, target_size=(256, 256)):
    image = load_img(image_path, color_mode='grayscale')
    label = load_img(label_path, color_mode='grayscale')
    image = img_to_array(image)
    label = img_to_array(label)
    
    # Compute padding
    h, w = image.shape[:2]
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    
    # Add padding
    image = np.pad(image, ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w), (0, 0)), mode='constant', constant_values=0)
    label = np.pad(label, ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w), (0, 0)), mode='constant', constant_values=0)
    
    # Resize to the target size
    image_resized = tf.image.resize(image, target_size) / 255.0
    label_resized = tf.image.resize(label, target_size) / 255.0
    label_resized = tf.cast(label_resized > 0.5, tf.float32)  # Binary mask
    return image_resized, label_resized

# Load and preprocess the test dataset
def load_and_process_data(image_dir, label_dir, target_size=(256, 256)):
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
    images, labels = [], []
    for img_path, lbl_path in zip(image_paths, label_paths):
        image, label = preprocess_image(img_path, lbl_path, target_size)
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

# Load the test images and labels
test_images, test_labels = load_and_process_data(image_dir, label_dir)

# Add batch dimension for prediction
test_images_input = np.expand_dims(test_images, axis=-1)

# Predict the masks for the test dataset
predicted_masks = model.predict(test_images_input)

# Flatten the arrays for accuracy computation
test_labels_flat = test_labels.flatten()
predicted_masks_flat = (predicted_masks.flatten() > 0.5).astype(np.float32)  # Threshold the predicted masks

# Compute accuracy
accuracy = accuracy_score(test_labels_flat, predicted_masks_flat)
print(f"Model Accuracy: {accuracy:.4f}")

# Optionally, plot a few results for visualization
n_samples = 7  # Number of samples to display
plt.figure(figsize=(15, 5))

for i in range(n_samples):
    plt.subplot(n_samples, 3, 3*i + 1)
    plt.imshow(test_images[i].squeeze(), cmap='gray')
    plt.title(f"Test Image {i+1}")
    plt.axis('off')

    plt.subplot(n_samples, 3, 3*i + 2)
    plt.imshow(test_labels[i].squeeze(), cmap='gray')
    plt.title(f"Ground Truth {i+1}")
    plt.axis('off')

    plt.subplot(n_samples, 3, 3*i + 3)
    plt.imshow(predicted_masks[i].squeeze(), cmap='jet')
    plt.title(f"Predicted Mask {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()
