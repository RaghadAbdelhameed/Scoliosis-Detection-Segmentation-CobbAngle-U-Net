import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Load the saved model
model = load_model('spine_rec_model_with_dropout_earlystop.h5')

# Load the test dataset
img_width = 25
img_height = 50
batch_size = 50

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    './test',
    labels='inferred',
    label_mode="int",
    class_names=['Normal', 'Scol'],
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_width, img_height),
    shuffle=False
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=2)

# Extract true labels and images
true_labels = np.concatenate([y.numpy() for x, y in test_dataset], axis=0)
images = np.concatenate([x.numpy() for x, y in test_dataset], axis=0)

# Make predictions
predictions = model.predict(test_dataset)
predicted_labels = np.argmax(predictions, axis=1)  # Convert logits to class indices

# Count correct and incorrect predictions
correct_predictions = np.sum(predicted_labels == true_labels)
incorrect_predictions = np.sum(predicted_labels != true_labels)

# Accuracy
total_images = len(true_labels)
accuracy = correct_predictions / total_images * 100

# Iterate through each image to display details
print("\nDetailed Results:")
for i in range(len(true_labels)):
    truth = "Normal" if true_labels[i] == 0 else "Scol"
    predicted = "Normal" if predicted_labels[i] == 0 else "Scol"
    result = "Correct" if true_labels[i] == predicted_labels[i] else "Incorrect"
    print(f"Image {i + 1}: Truth = {truth}, Predicted = {predicted}, Result = {result}")

# Summary statistics
print("\nSummary Statistics:")
print(f"Total Images: {total_images}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {incorrect_predictions}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Plot the results
categories = ['Correct Predictions', 'Incorrect Predictions']
counts = [correct_predictions, incorrect_predictions]

plt.bar(categories, counts, color=['green', 'red'])
plt.ylabel('Number of Images')
plt.title('Prediction Results')

# Add values on top of bars
for i, count in enumerate(counts):
    plt.text(i, count + 1, str(count), ha='center', fontsize=12)

plt.show()

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
labels = ['Normal', 'Scol']

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
