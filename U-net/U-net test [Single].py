import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import random

# Placeholder for paths
model_path = "D:/Reseach day/ARC/U-net/spinal_cord_unet.h5"  # Path to the saved model
image_dir = "D:/Reseach day/ARC/Processed_Scol(test)"  # Directory containing test images
label_dir = "D:/Reseach day/ARC/Processed_Scol(test)Line"  # Directory containing label images
output_dir = "D:/Reseach day/ARC/predictions"  # Directory to save the predicted masks

# Load the trained model
model = load_model(model_path)

# Image preprocessing function (padding and resizing)
def preprocess_image(image_path, label_path=None, target_size=(256, 256)):
    image = load_img(image_path, color_mode='grayscale')
    image = img_to_array(image)
    
    # Optional: Load label if provided
    label = None
    if label_path:
        label = load_img(label_path, color_mode='grayscale')
        label = img_to_array(label)
    
    # Compute padding
    h, w = image.shape[:2]
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    
    # Add padding
    image = np.pad(image, ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w), (0, 0)), mode='constant', constant_values=0)
    if label is not None:
        label = np.pad(label, ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w), (0, 0)), mode='constant', constant_values=0)
    
    # Resize to the target size
    image_resized = tf.image.resize(image, target_size) / 255.0
    label_resized = None
    if label is not None:
        label_resized = tf.image.resize(label, target_size) / 255.0
        label_resized = tf.cast(label_resized > 0.5, tf.float32)  # Binary mask
    
    return np.expand_dims(image_resized, axis=-1), label_resized  # Add channel dimension

# Predict and save the single image mask
def predict_and_save_single_image(image_path, label_path=None, output_dir="predictions"):
    # Preprocess image and label
    image, label = preprocess_image(image_path, label_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Predict the mask
    predicted_mask = model.predict(np.expand_dims(image, axis=0))[0]  # Add batch dimension
    predicted_mask_binary = (predicted_mask > 0.5).astype(np.float32)  # Thresholding the mask
    
    # Save the predicted mask
    save_path = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", "_predicted_mask.png"))
    plt.imsave(save_path, predicted_mask_binary.squeeze(), cmap='gray')
    print(f"Predicted mask saved to: {save_path}")
    
    # Plot original image, ground truth (optional), and predicted mask
    plt.figure(figsize=(12, 4))
    
    # Plot the original image
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    # Plot the ground truth label (if provided)
    if label is not None:
        plt.subplot(1, 3, 2)
        plt.imshow(label.numpy().squeeze(), cmap='gray')  # Convert to NumPy array if it's a tensor
        plt.title("Ground Truth")
        plt.axis('off')
    
    # Plot the predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask_binary.squeeze(), cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Get all image paths from the directory
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
label_paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".jpg")]

# Randomly select 10 images and corresponding labels
random_indices = random.sample(range(len(image_paths)), 10)
selected_image_paths = [image_paths[i] for i in random_indices]
selected_label_paths = [label_paths[i] for i in random_indices]

# Run the function for each of the selected images
for image_path, label_path in zip(selected_image_paths, selected_label_paths):
    predict_and_save_single_image(image_path, label_path, output_dir)
