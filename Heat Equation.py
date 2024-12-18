import numpy as np  # For handling arrays and numerical operations
import cv2  # For reading the image
import matplotlib.pyplot as plt  # For displaying images

# Parameters
K = 0.1  # Diffusivity coefficient (controls edge preservation)
alpha = 0.03  # Diffusion coefficient (controls smoothing intensity)
dt = 0.1  # Time step (smaller values improve stability)
num_steps = 50  # Number of timesteps (how long diffusion runs)

# Load the image from the specified directory
image_path = r".\Train\Scol\idiopathic-scoliosis-2 (1).jpg"  # Use raw string for Windows path
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

        # Anisotropic diffusivity function
        diffusivity = 1 / (1 + (grad_magnitude / K)**2)  # Update this with the anisotropic diffusivity formula

        # Compute the Laplacian (second derivatives)
        laplacian = (
            np.roll(u, 1, axis=0)
            + np.roll(u, -1, axis=0)
            + np.roll(u, 1, axis=1)
            + np.roll(u, -1, axis=1)
            - 4 * u
        )

        # Update the image according to the heat equation with edge preservation
        u = u + alpha * dt * diffusivity * laplacian

    # Scale the result back to [0, 255] for saving
    u_new_scaled = (u * 255).astype(np.uint8)

    # Histogram Equalization to enhance visibility
    u_new_equalized = cv2.equalizeHist(u_new_scaled)

    # Save the smoothed image as "processed_image.jpg" in the specified directory
    output_path = r"D:\Reseach day\ARC\heat_result.jpg"  # Specify the output file name and path
    cv2.imwrite(output_path, u_new_equalized)
    print(f"Smoothed image saved as: {output_path}")





image_path = "./Train/Scol/IMG-20241120-WA0033.jpg"  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not load the image.")
else:
    # Step 1: Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Experiment with clipLimit
    equalized = clahe.apply(image)
    
    # Step 2: Sobel Filter (Gradient Magnitude)
    grad_x = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)
    grad_magnitude = np.uint8(np.absolute(grad_magnitude))

    # Step 3: Thresholding for Spinal Cord Region
    _, thresholded = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)

    # Step 4: Region Growing (Seed-based Segmentation)
    # Start from a point near the spinal cord center (you can adjust this)
    seed_point = (image.shape[1] // 2, image.shape[0] // 2)
    mask = np.zeros_like(thresholded)
    cv2.floodFill(thresholded, mask, seed_point, 255)
    
    # Step 5: Morphological Operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    # Step 6: Final Mask Display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(equalized, cmap='gray')
    plt.title("Preprocessed Image (CLAHE)")

    plt.subplot(1, 2, 2)
    plt.imshow(cleaned_mask, cmap='gray')
    plt.title("Final Segmentation (Spinal Cord Mask)")

    plt.show()

    # Optionally save the final mask
    output_path = "spinal_cord_segmentation_result.png"
    cv2.imwrite(output_path, cleaned_mask)
    print(f"Segmentation saved to: {output_path}")
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Step 1: Histogram Equalization (CLAHE)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# equalized = clahe.apply(image)

# # Step 2: Smoothing with Gaussian Blur
# blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

# # Step 3: Edge Detection (Canny)
# edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# # Step 4: Morphological Operations
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
# opened_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, kernel)

# # Step 5: Contour Filtering (Optional)
# contours, _ = cv2.findContours(opened_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# mask = np.zeros_like(edges)
# for contour in contours:
#     # Filter based on area (adjust thresholds as needed)
#     if 500 < cv2.contourArea(contour) < 5000:
#         cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

# # Display Results
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 3, 1)
# plt.imshow(image, cmap='gray')
# plt.title("Original Image")
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.imshow(equalized, cmap='gray')
# plt.title("Enhanced Contrast")
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.imshow(mask, cmap='gray')
# plt.title("Refined Pseudo-Labels")
# plt.axis("off")

# plt.show()

# # image_path = ".\Train\Scol\IMG-20241109-WA0116.jpg"  # Replace with your file path
# # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # # Step 1: Apply Gaussian Blur for Smoothing
# # blurred = cv2.GaussianBlur(image, (5, 5), 0)

# # # Step 2: Apply Canny Edge Detection
# # edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

# # # Step 3: Morphological Operations to Refine
# # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# # refined_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# # # Display Results
# # plt.figure(figsize=(10, 5))
# # plt.subplot(1, 2, 1)
# # plt.imshow(image, cmap='gray')
# # plt.title("Preprocessed Image")
# # plt.axis("off")

# # plt.subplot(1, 2, 2)
# # plt.imshow(refined_edges, cmap='gray')
# # plt.title("Pseudo-Labels (Edges)")
# # plt.axis("off")

# # plt.show()







# # Load the processed image (after heat equation)
# processed_image = cv2.imread("heat_result.jpg", cv2.IMREAD_GRAYSCALE)

# # Apply Canny edge detection to create a mask
# edges = cv2.Canny(processed_image, threshold1=100, threshold2=200)

# # Show the edges on the original image
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(processed_image, cmap='gray')
# plt.title("Processed Image")
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(edges, cmap='gray')
# plt.title("Edge Detection Mask")
# plt.axis('off')

plt.show()
    # Display original and processed images side by side
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap="gray")
    # plt.title("Original Image")
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.imshow(u_new_equalized, cmap="gray")
    # plt.title("Processed Image")
    # plt.axis("off")

    # plt.show()
