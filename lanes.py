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

def display_lines(image, lines):
    line_image = np.zeros_like(image) # Black image
    if lines is not None:
        for line in lines:
            # Right now each line is 2D array [x1,y1,x2,y2]
            # Reshape it into 1D array with 4 elements
            x1, y1, x2, y2 = line.reshape(4)
            # Draw lines
            # cv2.line(input image, start point, end point, color, thickness)
            cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 5)
    return line_image

# Function to create white triangle and black mask of the input image.
# Apply AND bitwise operation on those 2 to get area of interest (lanes)
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
    # Isolate area of interest and fill rest with zeros (black)
    # Preform AND operation on elements from 2 arrays (image, mask)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Load image, make copy of it
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

canny = canny(lane_image)
cropped_image = region_of_interest(canny)
# https://bit.ly/2XQgX7I
# cv2.HoughLinesP(input image, rho, theta, treshold, placeholder array, min length of line, max allowed gap btw lines)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
line_image = display_lines(lane_image, lines)

# Blend image to another with defined weights(transparency, translucency)
# https://bit.ly/3iy2gxN
# cv2.addWeighted(original image, alpha, second image, beta, gamma)
combo_image = cv2.addWeighted(lane_image, 0.7, line_image, 1, 1)

# Display result image
cv2.imshow("Result", combo_image)
cv2.waitKey(0)
