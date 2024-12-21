import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_cobb_angle(image_path):
    # Load the binary mask image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Invert the image (curve becomes white, background black)
    inverted_image = cv2.bitwise_not(image)

    # Find contours
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Get the largest contour (spinal centerline)
    contour = max(contours, key=cv2.contourArea).squeeze()

    # Extract x and y coordinates
    x = contour[:, 0]
    y = contour[:, 1]

    # Compute the tangent slopes (dy/dx) using forward differences
    dx = np.diff(x, append=x[-1])
    dy = np.diff(y, append=y[-1])
    slopes = dy / (dx + 1e-6)  # Avoid division by zero

    # Compute the tangent slope at evenly spaced points
    num_points = 19
    indices = np.linspace(0, len(slopes) - 1, num_points, dtype=int)
    tangent_slopes = slopes[indices]
    selected_points = contour[indices]

    # Identify the points with maximum and minimum slopes
    max_slope_idx = np.argmax(tangent_slopes)
    min_slope_idx = np.argmin(tangent_slopes)

    p_max = selected_points[max_slope_idx]
    p_min = selected_points[min_slope_idx]

    # Calculate the Cobb angle
    max_slope = tangent_slopes[max_slope_idx]
    min_slope = tangent_slopes[min_slope_idx]

    cobb_angle = np.abs(np.degrees(np.arctan(max_slope) - np.arctan(min_slope)))

    # Visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')

    # Overlay the +x and +y axes
    plt.axhline(y=image.shape[0] // 2, color='green', linestyle='--', label='+x Axis')
    plt.axvline(x=image.shape[1] // 2, color='purple', linestyle='--', label='+y Axis')

    # Plot the centerline and selected points
    plt.plot(x, y, color='yellow', linewidth=2, label='Centerline')
    plt.scatter(selected_points[:, 0], selected_points[:, 1], color='cyan', label='Selected Points')
    plt.scatter(p_max[0], p_max[1], color='red', label='Max Slope Point')
    plt.scatter(p_min[0], p_min[1], color='blue', label='Min Slope Point')

    # Add annotations
    plt.text(p_max[0], p_max[1], "Max Slope", color='red')
    plt.text(p_min[0], p_min[1], "Min Slope", color='blue')
    plt.title(f'Cobb Angle: {cobb_angle:.2f} degrees')

    plt.legend()
    plt.axis('off')
    plt.show()

# Example usage
compute_cobb_angle('./predictions/predicted_mask.png')  # Replace with your image path
