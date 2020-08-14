"""
https://github.com/thladnik/Generic3DAnnotator/particle_detection.py - Automatic particle detection.
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
import h5py
import numpy as np

import gv
import processing

def run():

    ### Delete datasets if they exist (overwrite)
    if gv.KEY_PARTICLES in gv.f:
        print('Delete previous dataset \'{}\''.format(gv.KEY_PARTICLES))
        del gv.f[gv.KEY_PARTICLES]
    if gv.KEY_PART_CENTR in gv.f:
        print('Delete previous dataset \'{}\''.format(gv.KEY_PART_CENTR))
        del gv.f[gv.KEY_PART_CENTR]
    if gv.KEY_PART_AREA in gv.f:
        print('Delete previous dataset \'{}\''.format(gv.KEY_PART_AREA))
        del gv.f[gv.KEY_PART_AREA]

    ### Set image dataset
    dset = gv.dset

    max_part_num = 200
    ### Create particle datasets
    dset_part = gv.f.create_dataset(gv.KEY_PARTICLES,
                         shape=(dset.shape[0],max_part_num,50,2),
                         dtype=np.float64,
                         fillvalue=np.nan)
    dset_centr = gv.f.create_dataset(gv.KEY_PART_CENTR,
                         shape=(dset.shape[0],max_part_num,2),
                         dtype=np.float64,
                         fillvalue=np.nan)
    dset_area = gv.f.create_dataset(gv.KEY_PART_AREA,
                         shape=(dset.shape[0],max_part_num),
                         dtype=np.float64,
                         fillvalue=np.nan)

    gv.statusbar.startProgress('Detecting particles...', dset.shape[0])

    print('Start particle detection')
    for i in range(dset.shape[0]):
        if i % 50 == 0:
            gv.statusbar.setProgress(i+1)

        ### Squeeze discards contours with just 1 point automatically
        thresh_rule = gv.w.gb_part_detect.thresh_rule.currentText()
        std_mult = gv.w.gb_part_detect.std_mult.value()
        cnts = processing.particle_detector(np.rot90(dset[i, :, :, :], gv.f.attrs[gv.KEY_ROT]), thresh_rule, std_mult)
        #cnts = [c.squeeze() for c in cnts]

        if len(cnts) > dset_part.shape[1]:
            print('WARNING: too many contours. All discarded for frame {}'.format(i))
            continue


        ### Add contours to dataset
        for j, cnt in enumerate(cnts):
            if cnt.shape[0] > dset_part.shape[2]:
                print('WARNING: too many points for contour {} in frame {}. DISCARDED.'.format(j, i))
                continue


            ### Set contour centroid
            M = cv2.moments(cnt)
            dset_centr[i, j, :] = [M['m10']/M['m00'], M['m01']/M['m00']]

            ### Set contour area
            dset_area[i,j] = M['m00']

            ### Set contour points data
            dset_part[i,j, :cnt.shape[0],:] = cnt
        #print(i, dset_part[i,:cnts.shape[0], :cnts.shape[1],:cnts.shape[2]])

    gv.statusbar.endProgress()

    print('Particle detection finished')


