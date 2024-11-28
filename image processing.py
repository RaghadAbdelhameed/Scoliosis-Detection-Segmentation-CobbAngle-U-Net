import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
import os
from skimage.metrics import structural_similarity as ssim

# === PARAMETERS ===
start_time = time.time()
alpha = 0.03  # Diffusion coefficient
dt = 0.1  # Time step
num_steps = 25  # Number of timesteps

# === IMAGE PREPROCESSING ===
def load_images_from_folder(folder, label, image_size=(224, 224)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, image_size)
            images.append(img_resized)
            labels.append(label)
    return images, labels

# Dataset paths
normal_path = "C:/Users/pc/ARC/ARC/Dataset_Scoliosis/Normal/"
scoliosis_path = "C:/Users/pc/ARC/ARC/Dataset_Scoliosis/Scoliosis/"

normal_images, normal_labels = load_images_from_folder(normal_path, 0)
scoliosis_images, scoliosis_labels = load_images_from_folder(scoliosis_path, 1)

# Combine and normalize images
X = np.array(normal_images + scoliosis_images) / 255.0
X = X.reshape(-1, X.shape[1], X.shape[2], 1)  # For CNN input
y = np.array(normal_labels + scoliosis_labels)
y = to_categorical(y, 2)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === HEAT EQUATION SIMULATION ===
image_path = "D:/ff.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Error: Could not load image. Please check the file path: {image_path}")
    exit()

u = image / 255.0  # Normalize to [0, 1]
u_new = u.copy()

# Heat equation simulation using vectorized operations
for _ in range(num_steps):
    u_old = u_new.copy()
    gradient_x = u_old[2:, 1:-1] - u_old[:-2, 1:-1]
    gradient_y = u_old[1:-1, 2:] - u_old[1:-1, :-2]
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    diffusivity = np.exp(-(gradient**2) / (2 * 0.1**2))
    u_new[1:-1, 1:-1] = u_old[1:-1, 1:-1] + alpha * dt * diffusivity * (
        (u_old[2:, 1:-1] - 2 * u_old[1:-1, 1:-1] + u_old[:-2, 1:-1]) +
        (u_old[1:-1, 2:] - 2 * u_old[1:-1, 1:-1] + u_old[1:-1, :-2])
    )

u_new_scaled = (u_new * 255).astype(np.uint8)
u_new_equalized = cv2.equalizeHist(u_new_scaled)

output_path = "D:/DSA-SBME27/test1.jpg"
cv2.imwrite(output_path, u_new_equalized)

# SSIM comparison
ssim_value = ssim(image, u_new_equalized)
print(f"SSIM between original and processed image: {ssim_value}")

# === CORRECT COBB ANGLE CALCULATION ===
def calculate_cobb_angle(image):
    # Apply edge detection
    edges = cv2.Canny(image, 50, 150)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is None:
        print("No lines detected.")
        return None

    # Store line endpoints
    line_coords = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_coords.append(((x1, y1), (x2, y2)))

    # Sort lines based on their x-coordinates to find the most vertical lines (spine curvature)
    line_coords.sort(key=lambda coord: coord[0][0])  # Sort by x-coordinate of the first point

    # Ensure that two valid lines are selected for Cobb angle calculation
    if len(line_coords) >= 2:
        line1 = line_coords[0]  # First line (top vertebra)
        line2 = line_coords[-1]  # Last line (bottom vertebra)

        # Calculate the slopes of the two most tilted lines
        slope1 = (line1[1][1] - line1[0][1]) / (line1[1][0] - line1[0][0]) if (line1[1][0] - line1[0][0]) != 0 else 0
        slope2 = (line2[1][1] - line2[0][1]) / (line2[1][0] - line2[0][0]) if (line2[1][0] - line2[0][0]) != 0 else 0

        # Calculate the Cobb angle
        angle_rad = np.arctan(abs((slope1 - slope2) / (1 + slope1 * slope2)))
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    else:
        print("Insufficient lines for Cobb angle calculation.")
        return None

cobb_angle = calculate_cobb_angle(u_new_equalized)
if cobb_angle:
    print(f"Estimated Cobb Angle: {cobb_angle:.2f} degrees")
else:
    print("Cobb angle could not be calculated. Ensure proper line detection.")

# === CNN MODEL TRAINING ===
input_image = u_new_equalized.reshape(1, u_new_equalized.shape[0], u_new_equalized.shape[1], 1)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=10)

# === DETECTION AND VISUALIZATION ===
# Resize and preprocess the input image
input_image_resized = cv2.resize(input_image[0], (224, 224))  # Resize to 224x224
input_image_resized = input_image_resized.reshape(1, 224, 224, 1)  # Add batch dimension
input_image_resized = input_image_resized / 255.0  # Normalize if needed

# Predict scoliosis on the resized image
prediction = model.predict(input_image_resized)

scoliosis_detected = np.argmax(prediction)
detection_result = "Scoliosis Detected" if scoliosis_detected == 1 else "Normal"

# Visualization of all results
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Processed Image
plt.subplot(1, 4, 2)
plt.imshow(u_new_equalized, cmap='gray')
plt.title("Processed Image")
plt.axis('off')

# Scoliosis Detection
plt.subplot(1, 4, 3)
plt.imshow(u_new_equalized, cmap='gray')
plt.title(detection_result)
plt.axis('off')

# Cobb Angle
plt.subplot(1, 4, 4)
plt.imshow(u_new_equalized, cmap='gray')
plt.title(f"Cobb Angle: {cobb_angle:.2f}°" if cobb_angle else "Cobb Angle Calculation Failed")
plt.axis('off')

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")

plt.show()

