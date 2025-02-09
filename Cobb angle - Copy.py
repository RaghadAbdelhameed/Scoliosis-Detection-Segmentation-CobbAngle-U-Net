import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from skimage.morphology import closing, dilation, skeletonize

def preprocess_mask(image):
    """Apply morphological operations to close gaps in the spinal cord segmentation."""
    closed = closing(image, np.ones((5, 5), np.uint8))  # Close small gaps
    dilated = dilation(closed, np.ones((3, 3), np.uint8))  # Thicken the structure slightly
    return dilated

def extract_skeleton(mask):
    """Extract the skeleton of the segmented spinal cord."""
    skeleton = skeletonize(mask > 0)  # Skeletonization ensures a thin, centered representation
    return skeleton.astype(np.uint8) * 255  # Convert back to binary image

def extract_centerline(mask):
    """Find the largest connected component's skeleton."""
    skeleton = extract_skeleton(mask)
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).squeeze()

def fit_spline(x, y):
    """Fit a robust spline to the extracted centerline."""
    if len(x) < 4:  # Ensure we have enough points
        return x, y
    spline = UnivariateSpline(x, y, s=len(x) * 0.1)  # Adaptive smoothing
    x_smooth = np.linspace(min(x), max(x), 500)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

def draw_smooth_centerline(image_path):
    """Extracts and draws a smooth centerline along the spinal cord."""
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return
    
    # Step 1: Preprocess the mask
    processed_mask = preprocess_mask(image)
    
    # Step 2: Extract the skeleton-based centerline
    centerline = extract_centerline(processed_mask)
    if centerline is None or len(centerline) < 10:
        print("No valid spinal cord centerline detected.")
        return
    
    # Extract x, y coordinates of the centerline
    x, y = centerline[:, 0], centerline[:, 1]
    
    # Step 3: Fit a smooth spline along the centerline
    x_smooth, y_smooth = fit_spline(x, y)
    
    # Step 4: Plot results
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.plot(x, y, color='yellow', linewidth=2, label='Original Centerline')
    plt.plot(x_smooth, y_smooth, color='red', linewidth=3, label='Smoothed Centerline')
    plt.axis('off')
    plt.legend()
    plt.show()

# Example usage
draw_smooth_centerline('./predictions/predicted_mask.png')  # Replace with actual image path
