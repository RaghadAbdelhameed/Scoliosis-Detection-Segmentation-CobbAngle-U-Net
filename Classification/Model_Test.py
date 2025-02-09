import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set parameters to match training
IMG_HEIGHT = 25  # Matches training image height
IMG_WIDTH = 50   # Matches training image width
MODEL_PATH = 'spine_rec_model_optimized.h5'

def load_and_preprocess_image(file_path):
    """Replicates the exact preprocessing used in training"""
    img = load_img(
        file_path,
        color_mode='grayscale',
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    return img_array

def main(test_dir):
    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Data collection
    filenames = []
    true_labels = []
    pred_probs = []
    
    # Process images
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                
                # Get true label from directory structure
                class_name = os.path.basename(root).lower()
                true_label = 1 if class_name == 'scol' else 0
                
                try:
                    # Identical preprocessing to training
                    img_array = load_and_preprocess_image(file_path)
                    
                    # Prediction
                    pred_prob = model.predict(img_array, verbose=0)[0][0]
                    
                    # Store results
                    filenames.append(file_path)
                    true_labels.append(true_label)
                    pred_probs.append(pred_prob)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    # Convert probabilities to class labels
    pred_labels = [1 if prob >= 0.5 else 0 for prob in pred_probs]
    
    # Generate statistics
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, 
                               target_names=['Normal', 'Scol']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)
    
    # Visualize confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Scol'],
                yticklabels=['Normal', 'Scol'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Additional metrics
    accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Misclassified samples
    misclassified = [filenames[i] for i in range(len(filenames)) 
                    if true_labels[i] != pred_labels[i]]
    print(f"\nNumber of misclassified samples: {len(misclassified)}")
    print("First 10 misclassified samples:" if misclassified else "No misclassified samples")
    for path in misclassified[:10]:
        print(f" - {path}")

if __name__ == "__main__":
    test_directory = './Train'  # Update with your test directory path
    main(test_directory)