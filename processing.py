import cv2
import numpy as np
from skimage.filters import threshold_otsu
from scipy import stats
from scipy.signal import convolve2d

import gv
from median_subtraction import median_norm


def median_norm_filter(image, med_range):
    dshape = gv.f[gv.KEY_ORIGINAL].shape
    i = gv.w.viewer.slider.value()

    start = i - med_range
    end = i + med_range
    if start < 0:
        end -= start
        start = 0

    if end > dshape[0]:
        start -= end - dshape[0]
        end = dshape[0]

    return median_norm(image, gv.f[gv.KEY_ORIGINAL][start:end, :, :, :])

def threshold_filter(image, thresh, maxval, ttype):
    _, thresh = cv2.threshold(image, thresh, maxval, getattr(cv2, ttype))

    return thresh


def adaptive_threshold_filter(image, maxval, amethod, ttype, block_size, c):
    thresh = cv2.adaptiveThreshold(image, maxval, getattr(cv2, amethod), getattr(cv2, ttype), block_size, c)

    return thresh

def particle_detector(image, thresh_rule, std_mult):

    ### Separate cell image from background with Otsu thresholding
    #cell = image > threshold_otsu(image, nbins_otsu)

    ### Filter brightest pixels
    #potential_centers = image > np.percentile(image[cell], percentile)
    #potential_centers = image > percentile

    if thresh_rule == '>':
        comp = np.greater
        op = np.add
    else:
        comp = np.less
        op = np.subtract
    #x = stats.norm.pdf(np.arange(-3, 3.01, 3.0), scale=1.0)
    x = np.array([1,1,1])/3
    kernel = x.reshape((1, -1)) * x.reshape((-1, 1))

    #image = convolve2d(image, kernel)
    #image = convolve2d(image.T, kernel)

    potential_centers = comp(image, op(np.mean(image), std_mult * np.std(image)))

    potential_centers = np.logical_or(image > np.mean(image) + std_mult * np.std(image), image < np.mean(image) - std_mult * np.std(image))

    ### Detect contours
    cnts, hier = cv2.findContours(potential_centers.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    ### Reverse sort contour indices by area
    areas = sorted([(cv2.contourArea(cnt), i) for i, cnt in enumerate(cnts)])[::-1]

    ### Filter all contours with > 2 contour points
    #cnts2 = [cnts[i] for a, i in areas if a > 0]

    cnts2 = [np.fliplr(cnts[i].squeeze()) for a, i in areas if a > 0]

    return cnts2

def particle_filter(image, *args):

    cnts = particle_detector(image[:,:,0], *args)

    cnts = [np.fliplr(cnt) for cnt in cnts]

    cv2.drawContours(image, cnts, -1, (255, 255, 255), 3)

    return image