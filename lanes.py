import cv2
import numpy as np

# Load image, make copy of it, convert it to grayscale

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

# To reduce image noise use Gaussian blur (from sharp image get smooth image)
# cv2.GaussianBlur(input image, kernel matrix, standard deviation)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny edge detection -> search for high value (color) difference between neighbour pixels i.e. edges
# cv2.Canny (input image, first treshold, second treshhold)
canny = cv2.Canny(blur, 50, 150)

# Display result image
cv2.imshow('Result', canny)
cv2.waitKey(0)
