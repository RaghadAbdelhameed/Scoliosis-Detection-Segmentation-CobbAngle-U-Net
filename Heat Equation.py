import numpy as np  # For handling arrays and numerical operations
import cv2  # For reading the image
import matplotlib.pyplot as plt  # For displaying images

# Parameters
alpha = 0.03  # Diffusion coefficient (controls smoothing intensity)
dt = 0.1  # Time step (smaller values improve stability)
num_steps = 50  # Number of timesteps (how long diffusion runs)

# Load the image from the specified directory
image_path = r"./Dataset_Scoliosis/Scoliosis/N1-Rt-TAIS-F-15-yrs_jpg.rf.e7d71f10e01dee49c9300535e95049bf.jpg"  # Use raw string for Windows path
print(f"Loading image from: {image_path}")  # Print the image path

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale image

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Could not load image. Please check the file path: {image_path}")
else:
    print("Image loaded successfully. Proceeding with heat equation simulation...")

    # Normalize pixel values to the range [0, 1]
    u = image / 255.0  # Normalize to [0, 1] to make calculations easier

    # Heat equation simulation loop
    for _ in range(num_steps):
        # Compute gradients in the x and y directions
        grad_x = np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)
        grad_y = np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)
        
        # Calculate the gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Diffusivity based on gradient magnitude
        diffusivity = np.exp(-(grad_magnitude**2) / (2 * 0.1**2))  # Adjust sigma as needed

        # Compute the Laplacian (second derivatives)
        laplacian = (
            np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
            np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) -
            4 * u
        )

        # Update the image according to the heat equation with edge preservation
        u = u + alpha * dt * diffusivity * laplacian

    # Scale the result back to [0, 255] for saving
    u_new_scaled = (u * 255).astype(np.uint8)

    # Histogram Equalization to enhance visibility
    u_new_equalized = cv2.equalizeHist(u_new_scaled)

    # Save the smoothed image as "processed_image.jpg" in the specified directory
    output_path = r"D:\ARC\heat_result.jpg"  # Specify the output file name and path
    cv2.imwrite(output_path, u_new_equalized)
    print(f"Smoothed image saved as: {output_path}")

    # Display original and processed images side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(u_new_equalized, cmap="gray")
    plt.title("Processed Image")
    plt.axis("off")

    plt.show()
