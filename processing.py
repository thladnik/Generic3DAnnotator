"""
https://github.com/thladnik/Generic3DAnnotator/processing.py - Processing functions to be called from GUI and worker processes.
Copyright (C) 2020 Tim Hladnik

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import cv2
import numpy as np

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


    potential_centers = np.logical_or(image > np.mean(image) + std_mult * np.std(image), image < np.mean(image) - std_mult * np.std(image))

    ### Detect contours
    cnts, hier = cv2.findContours(potential_centers.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    ### Reverse sort contour indices by area
    areas = sorted([(cv2.contourArea(cnt), i) for i, cnt in enumerate(cnts)])[::-1]

    cnts2 = [np.fliplr(cnts[i].squeeze()) for a, i in areas if a > 0]

    return cnts2

def particle_filter(image, *args):

    cnts = particle_detector(image[:,:,0], *args)

    cnts = [np.fliplr(cnt) for cnt in cnts]

    cv2.drawContours(image, cnts, -1, (255, 255, 255), 3)

    return image