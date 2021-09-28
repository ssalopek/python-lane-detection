import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to convert image to grayscale, apply gaussian blur and detect edges
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # To reduce image noise use Gaussian blur (from sharp image get smooth image)
    # cv2.GaussianBlur(input image, kernel matrix, standard deviation)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny edge detection -> search for high value (color) difference between neighbour pixels i.e. edges
    # cv2.Canny (input image, first treshold, second treshhold)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    # Define triangle with its verticies
    polygons = np.array([
    [(200, height), (1100, height), (550, 250)]
    ])
    # Return an array of zeros with the same shape and type as a given array(image)
    # i.e. make black image with the same size of input image
    mask = np.zeros_like(image)
    # On black mask apply white(255) triangle
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Load image, make copy of it
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

canny = canny(lane_image)
cropped_image = region_of_interest(canny)

# Display result image
cv2.imshow("Result", cropped_image)
cv2.waitKey(0)
