import numpy as np       # For handling arrays and numerical operations
import cv2               # For reading the image
import matplotlib.pyplot as plt  # For displaying images

# Parameters
alpha = 0.03            # Diffusion coefficient (controls smoothing intensity)
dt = 0.1                # Time step (smaller values improve stability)
num_steps = 50          # Number of timesteps (how long diffusion runs)

# Load the image from the specified directory
image_path = "D:/DSA-SBME27/N146, Rt TAIS, M, 16 Yrs"  # Change this path to your image file
print(f"Loading image from: {image_path}")  # Print the image path

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale image

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Could not load image. Please check the file path: {image_path}")
else:
    print("Image loaded successfully. Proceeding with heat equation simulation...")
    # Normalize pixel values to the range [0, 1]
    u = image / 255.0  # Normalize to [0, 1] to make calculations easier
    # Create a copy of the image to store updated values
    u_new = u.copy()
    rows, cols = u.shape  # Dimensions of the image

    # Heat equation simulation loop
    for _ in range(num_steps):
        # Create a copy of the image for this timestep
        u_old = u_new.copy()

        # Apply the heat equation for each pixel (ignoring borders)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Basic gradient for edge detection
                gradient = np.sqrt(
                    (u_old[i+1, j] - u_old[i-1, j])**2 +
                    (u_old[i, j+1] - u_old[i, j-1])**2
                )

                # Diffusivity based on the gradient
                diffusivity = np.exp(-(gradient**2) / (2 * 0.1**2))  # Adjust the sigma as needed

                # Update rule with edge preservation
                u_new[i, j] = u_old[i, j] + alpha * dt * diffusivity * (
                    (u_old[i+1, j] - 2*u_old[i, j] + u_old[i-1, j]) +
                    (u_old[i, j+1] - 2*u_old[i, j] + u_old[i, j-1])
                )

    # Scale the result back to [0, 255] for saving
    u_new_scaled = (u_new * 255).astype(np.uint8)

    # Histogram Equalization to enhance visibility
    u_new_equalized = cv2.equalizeHist(u_new_scaled)

    # Save the smoothed image as "test1.jpg" in the same directory
    output_path = "D:/DSA-SBME27/test1.jpg"  # Specify the output file name and path
    cv2.imwrite(output_path, u_new_equalized)
    print(f"Smoothed image saved as: {output_path}")

    # Display original and processed images side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(u_new_equalized, cmap='gray')
    plt.title("Processed Image")
    plt.axis('off')

    plt.show()
