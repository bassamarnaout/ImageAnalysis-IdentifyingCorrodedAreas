import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import statistics
import cv2

import os
from PIL import Image

import GLCM


# import the necessary packages
from helpers import pyramid
from helpers import sliding_window

import math


from skimage.color import rgb2hsv

from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.colors import hsv_to_rgb

import matplotlib.pyplot as plt
from matplotlib import colors


"""
###################################################################################################
# Function to convert number into string
# Switcher is dictionary data type here
###################################################################################################
"""
####################################################################################################
def nameOfImageFile(imageNumber):
    switcher = {

        "1": 'corrosion1.jpg',
        "2": 'corrosion2.jpg',
        "3": 'corrosion3.jpg',
        "4": 'corrosion4.jpg',
        "5": 'corrosion5.jpg',
        "6": 'corrosion6.jpg',
        "7": 'corrosion7.jpg',
        "8": 'corrosion8.jpg',
        "9": 'corrosion9.jpg',
        "10": 'white_glacier.jpg',
        "11": 'car3.jpg',
        "12": 'sample1.jpg',
        "13": 'sample2.jpg',
        "14": 'sample3.jpg',
        "15": 'sample4.jpg',
        "16": 'sample5.jpeg',
        "17": 'pipes1.jpeg',
        "18": 'lion.png',
    }

    # get() method of dictionary data type returns
    # value of passed argument if it is present
    # in dictionary otherwise second argument will
    # be assigned as default value of passed argument
    return switcher.get(imageNumber, "nothing")



"""
####################################################################################################
# Function: plotImageIntoRGBSpace
# It plots the image into RGB space
####################################################################################################
"""
def plotImageIntoRGBSpace(img):
    # Plotting the image on 3D plot
    r, g, b = cv2.split(img)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(
        r.flatten(),g.flatten(),b.flatten(), facecolors=pixel_colors, marker="."
    )
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()

    cv2.waitKey(0)


"""
###################################################################################################
# Function: show_images
# Display a list of images in a single figure with matplotlib.
#
# Input Parameter:
# ----------------
# images: List of np.arrays compatible with plt.imshow
# rows (Default = 1): Number of rows in figure (number of columns is
#                     set to np.ceil(n_images/float(rows))).

# titles: List of titles corresponding to each image. Must have
#         the same length as images.
####################################################################################################
"""
def show_images(images, rows=1, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, np.ceil(n_images / float(rows)), n + 1)
        height, width = image.shape[:2]
        # print image.ndim
        if (image.ndim == 2 or image.ndim == 3 ) and width >1: # plotting if image
            plt.imshow(image,interpolation = 'nearest' , cmap=cm.gray)
        elif image.ndim == 2 and width ==1: # plotting of Histogram
            plt.plot(image)


        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()




"""
###################################################################################################
function : create_image()
###################################################################################################
"""
# Create a new image with the given size
def create_image(width, height):
  # image = Image.new("RGB", (i, j), "white")
  image = np.zeros((height, width), dtype="uint8")
  return image



"""
###################################################################################################
function : get_pixel()
###################################################################################################
"""
# Get the pixel from the given image
def get_pixel(image, i, j):
  # Inside image bounds?
  height = image.shape[0]  # Height if the image
  width = image.shape[1]  # Width of the image
  # width, height = image.size
  if i >= width or j >= height:
    return None

  # Get Pixel
  # pixel = image.getpixel((i, j))
  pixel = image[i,j]
  return pixel



"""
###################################################################################################
# Function: convert_grayscale
# Create a Grayscale version of the image.
#
# Input Parameter:
# ----------------
# image: colored image
# rows (Default = 1): Number of rows in figure (number of columns is
#                     set to np.ceil(n_images/float(rows))).
#
# Returns--neighbors : List of neighbors to the seed
####################################################################################################
"""
def convert_grayscale(image):
    # # Get size
    # width, height = image.size

    # grab the image dimensions
    height = image.shape[0]  # Height if the image
    width = image.shape[1]  # Width of the image

    # Create new Image and a Pixel Map
    grayscale_image = create_image(width, height)
    # pixels = grayscale_image.load()

    # Transform to grayscale
    for i in range(height - 1):
        for j in range(width - 1):
            if i == 537:
                print('stop')

            # Get Pixel
            # pixel = get_pixel(image, i, j)
            pixel = image[i, j]

            # Get R, G, B values (This are int from 0 to 255)
            red =   pixel[0]
            green = pixel[1]
            blue =  pixel[2]

            # Transform to grayscale
            gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)
            # gray = statistics.mean(pixel)
            # gray = (red + green + blue) / 3

            # Set Pixel in new image
            grayscale_image[i,j] = int(gray)
            # pixels[i, j] = (int(gray), int(gray), int(gray))

            # Return the grayscale image
    return grayscale_image




"""
###################################################################################################
function : averagePixels()
###################################################################################################
"""
# find the average rgb color for an image
def averagePixels(image):

    # grab the image dimensions
    height = image.shape[0]  # Height if the image
    width = image.shape[1]  # Width of the image

    r = g = b = count = 0
    for i in range(height-1):
        for j in range(width-1):
            # Get Pixel
            pixel = image[i, j]

            # Get      R, G, B values (This are int from 0 to 255)
            rTemp = pixel[0]
            gTemp = pixel[1]
            bTemp = pixel[2]

            r += rTemp
            g += gTemp
            b += bTemp

            count += 1

    # calculate averages
    print('average rgb color for an image, count of pixels')
    return (r / count), (g / count), (b/ count), count


"""
# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
# 1 images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, [img]
# 	2	channels : it is also given in square brackets. It the index of channel for which we calculate histogram.
# 	For example, if input is grayscale image, its value is [0]. For color image, you can pass [0],[1] or [2]
# 	to calculate histogram of blue,green or red channel respectively.
# 3	mask : mask image. To find histogram of full image, it is given as "None".
# But if you want to find histogram of particular region of image, you have to
# create a mask image for that and give it as mask. (I will show an example later.)
# 	4	histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
# 	5	ranges : this is our RANGE. Normally, it is [0,256].
#
"""
def getImageHistogram(image, bits_representation):
    hist = cv2.calcHist([image],[0],None,[2**bits_representation],[0,2**bits_representation])
    return hist




from PIL import Image
import matplotlib.pyplot as plt

def getRed(redVal):
    return '#%02x%02x%02x' % (redVal, 0, 0)

def getGreen(greenVal):
    return '#%02x%02x%02x' % (0, greenVal, 0)

def getBlue(blueVal):
    return '#%02x%02x%02x' % (0, 0, blueVal)



"""
###################################################################################################
function : plotHistogramOfAColoredImage()
###################################################################################################
"""
def plotHistogramOfAColoredImage(imageName):
    # Create an Image with specific RGB value
    image = Image.open(imageName)

    # Modify the color of two pixels

    image.putpixel((0 ,1), (1 ,1 ,5))

    image.putpixel((0 ,2), (2 ,1 ,5))

    # Display the image
    image.show()

    # Get the color histogram of the image
    histogram = image.histogram()

    # Take only the Red counts
    l1 = histogram[0:256]

    # Take only the Blue counts
    l2 = histogram[256:512]

    # Take only the Green counts
    l3 = histogram[512:768]

    plt.figure(0)

    # R histogram
    for i in range(0, 256):
        plt.bar(i, l1[i], color = getRed(i), edgecolor=getRed(i), alpha=0.3)

    # G histogram
    plt.figure(1)

    for i in range(0, 256):
        plt.bar(i, l2[i], color = getGreen(i), edgecolor=getGreen(i) ,alpha=0.3)

    # B histogram
    plt.figure(2)

    for i in range(0, 256):
        plt.bar(i, l3[i], color = getBlue(i), edgecolor=getBlue(i) ,alpha=0.3)

    plt.show()


import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy import misc
import scipy.misc


"""
###################################################################################################
function : plotHistogramOfHSVImage()
###################################################################################################
"""
def plotHistogramOfHSVImage(imageName):
    image_colored_rgb = scipy.misc.imread(imageName)
    array = np.asarray(image_colored_rgb)
    arr = (array.astype(float)) / 255.0
    img_hsv = colors.rgb_to_hsv(arr[..., :3])

    lu1 = img_hsv[..., 0].flatten()
    plt.subplot(1, 3, 1)
    plt.hist(lu1 * 360, bins=360, range=(0.0, 360.0), histtype='stepfilled', color='r', label='Hue')
    plt.title("Hue")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    lu2 = img_hsv[..., 1].flatten()
    plt.subplot(1, 3, 2)
    plt.hist(lu2, bins=100, range=(0.0, 1.0), histtype='stepfilled', color='g', label='Saturation')
    plt.title("Saturation")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    lu3 = img_hsv[..., 2].flatten()
    plt.subplot(1, 3, 3)
    plt.hist(lu3 * 255, bins=256, range=(0.0, 255.0), histtype='stepfilled', color='b', label='Intesity')
    plt.title("Intensity")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()




"""
###################################################################################################
function : sliding_window_()
###################################################################################################
"""
def sliding_window_(image_basic_RGB,image_basic_BGR, image_to_work_on):

    cv2.imshow('Original Image', image_basic_BGR)

    (winW, winH) = (15, 15)

    clone = image_basic_BGR.copy()

    hsv_img = rgb2hsv(image_basic_RGB)  # working fine

    hue_img = hsv_img[:, :, 0]
    saturation_img = hsv_img[:, :, 1]
    value_img = hsv_img[:, :, 2]

    testraster = clone.copy()

    # loop over the image pyramid
    for resized in pyramid(image_to_work_on, scale=0):

        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=winW, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		    # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		    # WINDOW

            [glcm, entropy, energy, contrast, homogeneity, dissimilarity,
             correlation] = GLCM.imageTextureMeasuresFromGLCM(window,
                                                              texture_type='none',
                                                              gray_levels=32,
                                                              displacement=[2, 2])

            if energy <= 0.05: #working
                # print('patch has been catched')
                # clone[y: y+window.shape[0], x:x+window.shape[1]] = 255

                for yy in range(y,y+window.shape[0]):
                    for xx in range(x,x+window.shape[1]):

                        h = hue_img[yy, xx]
                        s = saturation_img[yy, xx]
                        v = value_img[yy, xx]

                        # print('H Value: %f' % h)
                        # print('S Value: %f' % s)
                        # print('V Value: %f' % v)

                        if (not((v < 0.50) or (v > 0.50 and s < 0.20))) and h <= 0.07 and (resized[yy,xx] != 32 and resized[yy,xx] != 0):
                            HS = max(h,s)
                            if 0.30 <= HS <= 1.00:
                                clone[yy, xx] = (0, 0, 255)
                                testraster[yy, xx] = (0, 0, 255)
                            elif 0.15 <= HS < 0.30:
                                clone[yy, xx] = (0, 255, 0)
                                testraster[yy, xx] = (0, 255, 0)
                            elif 0.0 <= HS < 0.15:
                                clone[yy, xx] = (255, 0, 0)
                                testraster[yy, xx] = (255, 0, 0)

            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 165, 255), 0)
            cv2.namedWindow("RUST DEDECTION")
            cv2.imshow("RUST DEDECTION", clone)
            cv2.waitKey(1)
        # cv2.destroyAllWindows()

    # cv2.namedWindow("RUST DEDECTION...")
    # cv2.imshow('RUST DEDECTION', testraster)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    # return 0
    return testraster



"""
####################################################################################################
# THIS FUNCTION RESIZES THE IMAGE BY KEEPING ITS ASPECT RATIO
# INPUT : IMAGE, WIDTH, HIGHT
# RETURN VALUE: RESIZED IMAGE
####################################################################################################
"""
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized




"""
###################################################################################################
function : removeBackgroundImage()
###################################################################################################
"""
def removeBackgroundImage(grayImage):
    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format

    # -- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(grayImage, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    # -- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    # -- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
    img = img.astype('float32') / 255.0  # for easy blending

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
    masked = (masked * 255).astype('uint8')  # Convert back to 8-bit

    cv2.imshow('img', masked)  # Display
    cv2.waitKey()

