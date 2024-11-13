from skimage.metrics import structural_similarity as ssim
import cv2

# Load images (as grayscale for simplicity)
image1 = cv2.imread("./test1 before.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("./heat_result.jpg", cv2.IMREAD_GRAYSCALE)

# Calculate SSIM
score, diff = ssim(image1, image2, full=True)
print(f"SSIM Score: {score}")

# Optionally display the difference image
cv2.imshow("Difference", (diff * 255).astype("uint8"))
cv2.waitKey(0)
cv2.destroyAllWindows()
