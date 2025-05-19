import cv2
import numpy as np

# Load image
image = cv2.imread('jam.jpg')
if image is None:
    print("Failed to load image.")
    exit()
image = cv2.resize(image, (512, 512))

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Dark area range (HSV)
lower_dark = np.array([0, 0, 0])
upper_dark = np.array([180, 255, 50])

# Create mask
mask = cv2.inRange(hsv, lower_dark, upper_dark)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and calculate dark area
dark_area = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 50:
        dark_area += area
        cv2.drawContours(image, [cnt], -1, (0, 0, 255), 2)

# Decide quality
quality = "Good" if dark_area < 1000 else "Bad"
color = (0, 255, 0) if quality == "Good" else (0, 0, 255)
cv2.putText(image, f'Quality: {quality}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Display results
cv2.imshow('Dark Spot Mask', mask)
cv2.imshow('Fruit Inspection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()