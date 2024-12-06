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
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim

# === PARAMETERS ===
start_time = time.time()

alpha = 0.03  # Diffusion coefficient
dt = 0.1  # Time step
num_steps = 25  # Number of timesteps


# === IMAGE LOADING ===
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
normal_path = r".\Dataset_Scoliosis\Normal"
scoliosis_path = r".\Dataset_Scoliosis\Scoliosis"

normal_images, normal_labels = load_images_from_folder(normal_path, 0)
scoliosis_images, scoliosis_labels = load_images_from_folder(scoliosis_path, 1)

# Combine and normalize images
X = np.array(normal_images + scoliosis_images) / 255.0
X = X.reshape(-1, X.shape[1], X.shape[2], 1)  # For CNN input
y = np.array(normal_labels + scoliosis_labels)
y = to_categorical(y, 2)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# === COBB ANGLE CALCULATION ===
def calculate_cobb_angle(image):
    # Apply edge detection
    edges = cv2.Canny(image, 50, 150)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=15
    )
    if lines is None:
        print("No lines detected.")
        return None, None

    # Convert line coordinates to angles
    line_coords = []
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        if -80 < angle < -10 or 10 < angle < 80:  # Filter out horizontal/vertical lines
            line_coords.append(((x1, y1), (x2, y2)))
            angles.append(angle)

    if len(line_coords) < 2:
        print("Insufficient lines for Cobb angle calculation.")
        return None, None

    # Cluster angles to select two most distinct vertebral lines
    angles = np.array(angles).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(angles)
    cluster_centers = sorted(kmeans.cluster_centers_.flatten())

    # Calculate Cobb angle from the cluster centers
    angle_diff = abs(cluster_centers[1] - cluster_centers[0])
    cobb_angle = 180 - angle_diff  # Adjust for vertebral tilt
    return cobb_angle, line_coords


# === IMAGE PREPROCESSING ===
def heat_equation_simulation(image, alpha=0.03, dt=0.1, num_steps=25):
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
            (u_old[2:, 1:-1] - 2 * u_old[1:-1, 1:-1] + u_old[:-2, 1:-1])
            + (u_old[1:-1, 2:] - 2 * u_old[1:-1, 1:-1] + u_old[1:-1, :-2])
        )

    u_new_scaled = (u_new * 255).astype(np.uint8)
    u_new_equalized = cv2.equalizeHist(u_new_scaled)

    return u_new_equalized


# Process a single input image
image_path = r".\Dataset_Scoliosis\Normal\N2-N-F-17_1_0_jpg.rf.2da905b862064dbb91975bf08f501079.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Error: Could not load image. Please check the file path: {image_path}")
    exit()

u_new_equalized = heat_equation_simulation(image)

# === CNN MODEL TRAINING ===
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(2, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=10)

# Resize and preprocess the input image
input_image_resized = (
    cv2.resize(u_new_equalized, (224, 224)).reshape(1, 224, 224, 1) / 255.0
)

# Predict scoliosis on the resized image
prediction = model.predict(input_image_resized)
scoliosis_detected = np.argmax(prediction)
detection_result = "Scoliosis Detected" if scoliosis_detected == 1 else "Normal"

# === VISUALIZATION ===
plt.figure(figsize=(20, 8))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# Processed Image (After Heat Equation)
plt.subplot(1, 4, 2)
plt.imshow(u_new_equalized, cmap="gray")
plt.title("Processed Image")
plt.axis("off")

# Scoliosis Detection Result
plt.subplot(1, 4, 3)
plt.imshow(u_new_equalized, cmap="gray")
plt.title(f"Scoliosis Detection: {detection_result}")
plt.axis("off")

# Cobb Angle with Line Detection
plt.subplot(1, 4, 4)
output_image = cv2.cvtColor(u_new_equalized, cv2.COLOR_GRAY2BGR)

# Calculate Cobb Angle
cobb_angle, line_coords = calculate_cobb_angle(u_new_equalized)
if cobb_angle:
    for line in line_coords:
        cv2.line(output_image, line[0], line[1], (0, 255, 0), 2)

plt.imshow(output_image)
plt.title(
    f"Cobb Angle: {cobb_angle:.2f}°" if cobb_angle else "Cobb Angle Calculation Failed"
)
plt.axis("off")

# Show the plots
plt.show()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")
