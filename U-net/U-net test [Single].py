import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Configuration paths
model_path = "./U-net/spinal_cord_unet.h5"
image_path = "./Processed_Scol(test)/6.jpg"
label_path = "./Processed_Scol(test)Line/6.jpg"
output_dir = "./predictions"

# Load trained model
model = load_model(model_path)

def preprocess_image(image_path, label_path=None, target_size=(256, 256)):
    """Load and preprocess image with padding metadata tracking"""
    # Load original image
    image = load_img(image_path, color_mode='grayscale')
    image_array = img_to_array(image)
    original_h, original_w = image_array.shape[:2]
    
    # Calculate padding needed to make square
    max_dim = max(original_h, original_w)
    pad_h = (max_dim - original_h) // 2
    pad_w = (max_dim - original_w) // 2
    
    # Pad image to square
    padded_image = np.pad(image_array,
                         ((pad_h, max_dim - original_h - pad_h),
                          (pad_w, max_dim - original_w - pad_w),
                          (0, 0)),
                         mode='constant',
                         constant_values=0)
    
    # Resize and normalize
    processed_image = tf.image.resize(padded_image, target_size) / 255.0
    processed_image = np.expand_dims(processed_image, axis=-1)  # Add channel dimension
    
    # Process label if provided
    processed_label = None
    if label_path:
        label = load_img(label_path, color_mode='grayscale')
        label_array = img_to_array(label)
        
        # Pad label using same parameters
        padded_label = np.pad(label_array,
                             ((pad_h, max_dim - original_h - pad_h),
                              (pad_w, max_dim - original_w - pad_w),
                              (0, 0)),
                             mode='constant',
                             constant_values=0)
        
        # Resize and binarize label
        processed_label = tf.image.resize(padded_label, target_size) / 255.0
        processed_label = tf.cast(processed_label > 0.5, tf.float32).numpy()
    
    return (processed_image, (original_h, original_w, pad_h, pad_w)), processed_label

def predict_and_save_single_image(image_path, label_path=None, output_dir="predictions"):
    """Process image, predict mask, and save processed result"""
    # Preprocess with metadata
    (processed_image, metadata), processed_label = preprocess_image(image_path, label_path)
    original_h, original_w, pad_h, pad_w = metadata
    
    # Make prediction
    prediction = model.predict(np.expand_dims(processed_image, axis=0))[0]
    binary_mask = (prediction > 0.5).astype(np.float32)
    
    # Resize mask to original padded size
    max_dim = max(original_h, original_w)
    resized_mask = tf.image.resize(binary_mask, (max_dim, max_dim), method='nearest').numpy().squeeze()
    
    # Remove padding and rotate
    cropped_mask = resized_mask[pad_h:pad_h+original_h, pad_w:pad_w+original_w]
    rotated_mask = np.rot90(cropped_mask, k=-1)  # Clockwise 90 rotation
    
    # Save final mask
    os.makedirs(output_dir, exist_ok=True)
    plt.imsave(os.path.join(output_dir, "predicted_mask.png"), rotated_mask, cmap='gray')
    
    # Visualization
    plt.figure(figsize=(12, 4))
    
    # Preprocessed input
    plt.subplot(1, 3, 1)
    plt.imshow(processed_image.squeeze(), cmap='gray')
    plt.title("Network Input")
    plt.axis('off')
    
    # Ground truth (if available)
    if processed_label is not None:
        plt.subplot(1, 3, 2)
        plt.imshow(processed_label.squeeze(), cmap='gray')
        plt.title("True Mask")
        plt.axis('off')
    
    # Final prediction
    plt.subplot(1, 3, 3)
    plt.imshow(rotated_mask, cmap='gray')
    plt.title("Processed Prediction")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Execute prediction pipeline
predict_and_save_single_image(image_path, label_path, output_dir)