import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

def calculate_curvature(x, y):
    """ Calculate the curvature of the spline """
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature

def calculate_drift(y_original, y_smooth):
    """ Calculate the drift (max difference) between original contour and smoothed curve """
    drift = np.abs(y_original - y_smooth)
    max_drift = np.max(drift)
    return max_drift

def fill_gaps(x, y, gap_threshold=10):
    """ Fill the gaps in the contour by interpolating missing points between large gaps """
    sorted_indices = np.argsort(x)
    x_sorted = np.array(x)[sorted_indices]
    y_sorted = np.array(y)[sorted_indices]
    
    gap_indices = np.where(np.diff(x_sorted) > gap_threshold)[0]

    if len(gap_indices) == 0:
        return x_sorted, y_sorted
    
    new_x = []
    new_y = []
    
    new_x.extend(x_sorted[:gap_indices[0] + 1])
    new_y.extend(y_sorted[:gap_indices[0] + 1])
    
    for i in range(len(gap_indices)):
        x_start = x_sorted[gap_indices[i]]
        x_end = x_sorted[gap_indices[i] + 1]
        
        x_interpolated = np.linspace(x_start, x_end, num=10)
        y_interpolated = np.interp(x_interpolated, x_sorted, y_sorted)
        
        new_x.extend(x_interpolated)
        new_y.extend(y_interpolated)
        
    new_x.extend(x_sorted[gap_indices[-1] + 1:])
    new_y.extend(y_sorted[gap_indices[-1] + 1:])
    
    return np.array(new_x), np.array(new_y)

def calculate_slope(spline, x_vals):
    """ Calculate the slope of the tangent at each point using the first derivative of the spline """
    return spline.derivative()(x_vals)

def calculate_normal_vector(slope):
    """ Calculate the normal vector (perpendicular to the tangent) given the slope """
    normal_slope = -1 / slope if slope != 0 else float('inf')  # Perpendicular slope
    normal_vector = np.array([1, normal_slope])
    normal_vector /= np.linalg.norm(normal_vector)  # Normalize the vector
    return normal_vector

def intersection_of_lines(p1, v1, p2, v2):
    """ Calculate the intersection of two lines given by point and direction (p1, v1) and (p2, v2) """
    A = np.array([v1, -v2]).T
    b = p2 - p1
    t = np.linalg.solve(A, b)
    intersection = p1 + t[0] * v1
    return intersection

def angle_between_vectors(v1, v2):
    """ Calculate the acute angle between two vectors """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clipped to avoid numerical errors
    angle_deg = np.degrees(angle_rad)  # Convert from radians to degrees
    return angle_deg

def draw_smooth_centerline(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    inverted_image = cv2.bitwise_not(image)

    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).squeeze()

    x = contour[:, 0]
    y = contour[:, 1]

    x_filled, y_filled = fill_gaps(x, y)

    x_to_y = {}
    for i in range(len(x_filled)):
        if x_filled[i] not in x_to_y:
            x_to_y[x_filled[i]] = []
        x_to_y[x_filled[i]].append(y_filled[i])

    unique_x = sorted(x_to_y.keys())
    averaged_y = [np.mean(x_to_y[x_val]) for x_val in unique_x]

    max_y = np.max(averaged_y)
    drift_limit = 0.03 * max_y

    best_s = None
    min_curvature = float('inf')

    for s_candidate in range(1, 10000, 1):
        spline = UnivariateSpline(unique_x, averaged_y, s=s_candidate)
        y_smooth = spline(unique_x)

        curvature = calculate_curvature(unique_x, y_smooth)
        drift = calculate_drift(averaged_y, y_smooth)
        
        max_curvature = np.max(np.abs(curvature))
        
        if max_curvature < min_curvature and drift <= drift_limit:
            min_curvature = max_curvature
            best_s = s_candidate

    spline = UnivariateSpline(unique_x, averaged_y, s=best_s)
    x_smooth = np.linspace(min(unique_x), max(unique_x), 300)
    y_smooth = spline(x_smooth)

    # Choose 19 equally spaced points on the smoothed curve
    num_points = 19
    x_sample = np.linspace(min(x_smooth), max(x_smooth), num_points)
    y_sample = spline(x_sample)

    # Calculate the slopes (tangents) at each of the 19 points
    slopes = calculate_slope(spline, x_sample)

    # Find the indices of the maximum and minimum slopes
    max_slope_idx = np.argmax(slopes)
    min_slope_idx = np.argmin(slopes)

    # Calculate normal vectors at the max and min slope points
    normal_max = calculate_normal_vector(slopes[max_slope_idx])
    normal_min = calculate_normal_vector(slopes[min_slope_idx])

    # Plotting the results
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')

    plt.plot(x, y, color='yellow', linewidth=2, label='Original Centerline')
    plt.plot(x_smooth, y_smooth, color='red', linewidth=2, label=f'Smoothed Curve (s={best_s})')

    # Plot the points with maximum and minimum slopes
    plt.scatter(x_sample[max_slope_idx], y_sample[max_slope_idx], color='green', zorder=5, label='Max Slope')
    plt.scatter(x_sample[min_slope_idx], y_sample[min_slope_idx], color='blue', zorder=5, label='Min Slope')

    # Plot the normal vectors as arrows
    plt.arrow(x_sample[max_slope_idx], y_sample[max_slope_idx], 20 * normal_max[0], 20 * normal_max[1], 
              head_width=5, head_length=5, fc='green', ec='green', label='Normal at Max Slope')
    plt.arrow(x_sample[min_slope_idx], y_sample[min_slope_idx], 20 * normal_min[0], 20 * normal_min[1], 
              head_width=5, head_length=5, fc='blue', ec='blue', label='Normal at Min Slope')

    # Calculate intersection of the two normal lines
    normal_max_point = np.array([x_sample[max_slope_idx], y_sample[max_slope_idx]])
    normal_min_point = np.array([x_sample[min_slope_idx], y_sample[min_slope_idx]])

    intersection_point = intersection_of_lines(normal_max_point, normal_max, normal_min_point, normal_min)

    # Plot the intersection point
    plt.scatter(intersection_point[0], intersection_point[1], color='red', zorder=5, label='Intersection')

    # Calculate the angle between the normal vectors
    angle = angle_between_vectors(normal_max, normal_min)
    cobb_angle = 180 - angle

    print(f"Cobb angle: {cobb_angle:.2f} degrees")

    # Modify the legend to include the Cobb angle
    plt.title(f'Smoothed Centerline with Normal Vectors and Intersection')
    
    # Custom legend to add the Cobb angle in the box
    plt.legend(title=f'Cobb angle: {cobb_angle:.2f}°')


    plt.axis('off')
    plt.show()

# Example usage
draw_smooth_centerline('./predictions/predicted_mask.png')  # Replace with your image path