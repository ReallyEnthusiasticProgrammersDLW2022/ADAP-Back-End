import cv2
import numpy as np

def imageProcesser(img):
  imgGray = img.convert('L')
  return imgGray

def coordinateProcessor(coordinates):
  coordinates = coordinates.split(',')
  return float(coordinates[0]), float(coordinates[1])

def extractSegment(img, colour):
  diff = 10
  boundaries = [([colour[2], colour[1]-diff, colour[0]-diff],[colour[2]+diff, colour[1]+diff, colour[0]+diff])]
  for (lower, upper) in boundaries:
    # You get the lower and upper part of the interval:
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(img, lower, upper)

    # You can use the mask to count the number of white pixels.
    # Remember that the white pixels in the mask are those that
    # fall in your defined range, that is, every white pixel corresponds
    # to a green pixel. Divide by the image size and you got the
    # percentage of green pixels in the original image:
    ratio_green = cv2.countNonZero(mask)/(img.size/3)

    # This is the color percent calculation, considering the resize I did earlier.
    colorPercent = (ratio_green * 100)

    # Print the color percent, use 2 figures past the decimal point
    return colorPercent
