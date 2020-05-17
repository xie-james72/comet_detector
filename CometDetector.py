# initial goal to recreate original material
# overall goal to adapt the original material to a distributed system and identify areas of improvement
# http://www2.imm.dtu.dk/pubdb/edoc/imm5511.pdf
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares


# i'll likely need to create a class for the objects i detect to store their features
# and also a class for each image sequence


def contrastStretch(img, maxscale=255.0):
    # Performs a linear contrast stretch operation on a greyscale image by linearly scaling up the
    # intensity until the brightest pixel is at 255.
    img = cv2.multiply(img.astype('float32'), maxscale / np.max(img))
    return img.astype(np.uint8)


def remove_timestamp(img):
    # removes the timestamp found in the lower left-hand corner of all LASCO C2 and C3 images by
    # performing a connected component analysis on a ROI and removing the largest 16
    # there are always 16 objects in the timestamp (4 year, 2 mo, 2 day, 2 hour, 2 min, 2 slashes, colon (2 dots))

    # this fails if there is another large bright object in the ROI, though in the original paper those objects would
    # also have been removed
    ROI = img[490:511, 0:200]
    ROI_clean = cv2.threshold(ROI, 128, 1, cv2.THRESH_BINARY)[1]
    n, label = cv2.connectedComponents(ROI_clean)

    if n >= 16:
        objects = ndimage.find_objects(label)
        sizes = ndimage.sum(ROI, labels=label, index=range(n + 1))
        threshold = np.sort(sizes)[n - 16]
        for i, size in enumerate(sizes):
            if size >= threshold:
                ROI[objects[i]] = 0
    else:
        print('there are fewer than 16 objects in the ROI')

    img[490:511, 0:200] = ROI
    return img


def residuals(p, x, y):
    # returns the residuals of a histogram curve fit, given the parameters of the curve p, the domain x, and the
    # frequencies y
    a, b, c = p
    res = [f - (a * np.exp(b * i) - a * np.exp(c * i)) for i, f in zip(x, y)]
    return res


def cleanImage(imgA, imgB, sigma=0.7):
    # Removing noise from images via bandpass filter and increasing contrast
    # the high-pass filter is performed by subtracting two images of similar timestamp to remove stationary objects
    # the low-pass filter is a gaussian blur to preserve faint and small objects which may be removed in a median filter

    imgA = remove_timestamp(imgA)
    imgA = cv2.GaussianBlur(imgA, (3, 3), sigma)  # low pass filter
    imgB = cv2.GaussianBlur(imgB, (3, 3), sigma)  # low pass filter
    imgAB = cv2.subtract(imgA, imgB)  # high pass filter
    imgAB = contrastStretch(imgAB)  # min/max Contrast stretching; this might fuck with the histogram. Test it out.
    return imgAB


def find_maxima(img, h=1, kern=np.ones((3, 3), np.uint8)):
    # this function identifies regional maxima of an image using greyscale morphological reconstruction and returns a
    # boolean array of where these maxima are located
    # http://www2.vincent-net.com/luc/papers/93ieeeip_recons.pdf
    dilated = cv2.subtract(img, h)
    while True:
        # dilate the lower-intensity image
        new_dilated = cv2.dilate(dilated, kern, iterations=1)
        # check for where the dilated image intensity is greater than the original image intensity, then
        # set those pixels to the image intensity. This creates an image with 'cut peaks' in intensity
        # Iterate this process. Once this stabilizes, you will have an image with 'cut peaks
        indices = np.where(new_dilated > img)
        new_dilated[indices] = img[indices]
        if np.all(new_dilated == dilated):
            break
        else:
            dilated = new_dilated

    maxima = (img - new_dilated).astype(bool)
    return maxima


def remove_noise_floor(img, maxima, alpha=0.01):
    # this function receives an image and its local maxima, and outputs the intensity noise floor below which you should
    # ignore the maxima. This is done by creating a histogram of the maxima intensities, then fitting a gaussian-like
    # function with boundary conditions f -> 0 as I -> 0 and I -> 255

    intensity = img[maxima]
    hist = {i: sum(intensity[intensity == i]) for i in range(256)}
    # print(hist)
    # The histogrm takes the form of f(I) = a*exp(-bI) - a*exp(-cI), with parameters estimated using least squares
    p0 = np.array([10000.0, -0.5, -0.5])  # initial function parameters
    p = least_squares(residuals, p0, args=(list(hist.keys()), hist.values()), method='lm', verbose=1).x
    # print(p)
    # Find the histogram peak and from that calculate the intensity noise floor as the intensity with a fraction (alpha)
    # of the peak frequency, rounded down
    hist_fit = {i: p[0] * (np.exp(p[1] * i) - np.exp(p[2] * i)) for i in list(hist.keys())}
    # print(hist_fit)
    noise_floor_freq = max(hist_fit.values()) * alpha
    # print(noise_floor_freq)

    k = 0
    while hist_fit[k] <= noise_floor_freq:
        noise_floor = k
        k += 1
        # print(noise_floor)

    # The issue im encountering is that the image noise floor is 0
    # Maybe i'll try to reach out and ask about it....
    # For any object in the image with maxima less than the noise floor, delete it
    # cleanimg = cv2.subtract(img, noise_floor)
    return noise_floor


# Clean or discard the image
# Omit images with over a certain number of objects - this might be due to CMEs
# Identify all candidate objects in an image
# Items with super low variability are likely hot pixels.
# Record positions
# Correlate objects between images to determine a plausible path
# Remove stars. They will be moving horizontally from left to right at a near constant rate
# Figure out what that constant rate is
# For objects that dont move in the y much but move in the x by nearly the constant, thats a star. The "constant" needs to be pretty loose
# Remove irregularly moving objects
# the first pair, select all pairs of remaining objects within a reasonable distance (scaled with time?)
# In the third image, any object that "moves too fast" or is too far away should not be considered correlated
# Compare pairs from AB and BC
# All comets will be moving towards the sun
# Other parameters that should be relatively consistent between two images to ensure objects are similar are sharpness, elongation, and intensity
# Look into using k-d trees to make this a faster process

# Known issues in the algorithm:
# Rare comets that come close to earth will have highly irregular paths
# Comets which pass near stars are lost due to the blur and treated as a single object
# a suggestion was to just mark the object as a star for now rather than deleting???

mypath = 'C:/Users/James Xie/Desktop/c3/20200426/'
filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
'''
time = []
objects = []
max_time = 60
for f in filenames:  # assumes c3, 512 x 512 image, jpg
    time.append(int(f[9:-11]))
'''

A = cv2.imread(mypath + filenames[0], cv2.IMREAD_GRAYSCALE)
B = cv2.imread(mypath + filenames[1], cv2.IMREAD_GRAYSCALE)

AB = cleanImage(A, B)
ABmax = find_maxima(AB)
remove_noise_floor(AB, ABmax)
remove_timestamp(A)

cv2.imshow('figure_2', AB)
cv2.waitKey(0)
