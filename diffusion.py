import numpy as np  # For handling arrays and numerical operations
import cv2  # For reading the image
import matplotlib.pyplot as plt  # For displaying images
from skimage.metrics import structural_similarity as ssim  # Import SSIM
import time

start_time = time.time()


# Parameters
alpha = 0.01  # Diffusion coefficient (controls smoothing intensity)
dt = 0.1  # Time step (smaller values improve stability)
num_steps = 25  # Number of timesteps (how long diffusion runs)
kappa = 0.1  # Edge sensitivity parameter (smaller values make diffusivity more sensitive to edges)

# Load the image
image_path = "D:/IMG-20241114-WA0002.jpg"  # Change this path to your image file
print(f"Loading image from: {image_path}")

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale image
if image is None:
    print(f"Error: Could not load image. Please check the file path: {image_path}")
    exit()

print("Image loaded successfully. Proceeding with anisotropic diffusion simulation...")

# Normalize the image to the range [0, 1]
u = image / 255.0

# Anisotropic Diffusion Simulation
for _ in range(num_steps):
    # Calculate gradients
    grad_x = np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)  # Gradient along x
    grad_y = np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)  # Gradient along y

    # Compute gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Compute diffusivity function (edge-sensitive)
    diffusivity = 1 / (1 + (grad_magnitude / kappa) ** 2)

    # Calculate the divergence (Laplacian-like term) with diffusivity weighting
    diff_x = diffusivity * grad_x
    diff_y = diffusivity * grad_y

    # Update rule (divergence of diffusivity-weighted gradients)
    divergence = (
        np.roll(diff_x, 1, axis=0)
        - np.roll(diff_x, -1, axis=0)
        + np.roll(diff_y, 1, axis=1)
        - np.roll(diff_y, -1, axis=1)
    )
    u += alpha * dt * divergence

# Scale the result back to [0, 255] for saving
u_scaled = (u * 255).astype(np.uint8)

# Apply histogram equalization to enhance contrast
u_equalized = cv2.equalizeHist(u_scaled)

# Save the processed image with a specified filename
output_path = "D:/DSA-SBME27/test1.jpg"
cv2.imwrite(output_path, u_equalized)
        # SSIM comparison
ssim_value = ssim(image, u_equalized)
print(f"SSIM between original and processed image: {ssim_value}")
print(f"Processed image saved as: {output_path}")

# Display the original and processed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(u_equalized, cmap="gray")
plt.title("Processed Image")
plt.axis("off")
end_time = time.time()
execution_time = end_time - start_time
print(f'execution_time = {execution_time}')
plt.show()


