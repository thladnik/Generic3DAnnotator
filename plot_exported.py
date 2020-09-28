"""
https://github.com/thladnik/Generic3DAnnotator/plot_exported.py - Example script for plotting pickled position data.
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
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy import stats, interpolate, signal


def load_file(filepath, lowerbound, upperbound):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)

    positions = data['positions']
    times = data['time']

    ### Write to uniform array
    pos = np.nan * np.ones((len(positions), 50, 2))
    for i, c in enumerate(positions):
        for j, cc in enumerate(c):
            if cc[1] < lowerbound or cc[1] > upperbound:
                pos[i, j, :] = [np.nan, np.nan]
            else:
                pos[i, j, :] = cc

    return pos, times

def filter_position(times, pos, fps, cutoff, discard_factor, restitute_factor, plot=False):

    print('Input shape: {}'.format(pos.shape))
    new_pos = np.zeros(pos.shape[:2])
    print('Output shape: {}'.format(new_pos.shape))

    if plot:
        colors = plt.get_cmap('Dark2').colors
        plt.figure()

    for i in range(pos.shape[1]):
        print('Particle {}'.format(i))
        p = pos[:,i,1]

        if np.isnan(p).sum() > 0:
            print('WARNING: {} NaNs in position dataset. '.format(np.isnan(p).sum())
                  + 'Substituting with zeros for filtering')
            p[np.isnan(p)] = 0.

        ### Copy original position
        p_orig = np.copy(p)

        ### Plot before
        if plot:
            plt.plot(times, p, '-', color=colors[i], linewidth=.3, alpha=0.5)

        ### 1st pass: Discard positions based distance to lowpass filtered version
        ## Filter
        nyquist = fps/2
        b, a = signal.butter(1, cutoff/nyquist, 'low')
        p_filt = signal.filtfilt(b, a, p)

        ## Discard
        p_diff = np.abs(p - p_filt)
        p_diff_sd = np.nanstd(p_diff)
        p[p_diff > discard_factor*p_diff_sd] = np.nan

        if plot:
            plt.plot(times, p_filt, '--', color=colors[i], linewidth=2., alpha=0.3, label='Fish {}'.format(i))
            plt.plot(times, p, '-', color=colors[i], linewidth=2., alpha=1.0, label='Fish {}'.format(i))

        ### 2nd pass: Interpolate previously discarded positions and substitute
        #              with original based on distance of original/interpolation
        num_iters = 0
        num_changed = 1
        while num_changed > 0:
            num_changed = 0

            speed_sd = np.nanstd(np.diff(p))

            p_interp = interpolate.interp1d(times[np.isfinite(p)], p[np.isfinite(p)],
                                            'linear', fill_value=np.nan, bounds_error=False)(times)
            for j, pi in enumerate(p_interp):
                if np.isnan(p[j]) and np.abs(p_orig[j] - pi) < restitute_factor*speed_sd:
                    p[j] = p_orig[j]
                    num_changed += 1

            print('Iterations {}: {} changed'.format(num_iters, num_changed))

            num_iters += 1

        if plot:
            plt.plot(times, p, '-', color=colors[i], linewidth=1.2, alpha=1.0, label='Fish {}'.format(i))

        new_pos[:,i] = p

    if plot:
        plt.ylabel('Y-Position [cm]')
        plt.xlabel('Time [s]')
        plt.show()

    return new_pos

if __name__ == '__main__':

    #filepath = 'test_pickle/swimheight_light_5cm_Scale50pc.pickle' # Very good
    filepath = 'test_pickle/swimheight_dark_5cm_Scale100pc.pickle' # Good
    #filepath = 'test_pickle/swimheight_light_10cm_Scale100pc.pickle' # Okay
    #filepath = 'test_pickle/swimheight_dark_15cm_Scale100pc.pickle' # Terrible problems

    water_surface = 0.
    max_water_depth = 5.
    pos, times = load_file(filepath, water_surface, max_water_depth)

    fish_num = 6
    times = times[5:3000]
    pos = pos[5:3000,:fish_num,:]

    new_pos = filter_position(
                        times=times,
                        pos=pos,
                        fps=5,
                        cutoff=0.1,
                        discard_factor=0.5,
                        restitute_factor=5.0,
                        plot=True)




    data_out = dict()
    data_out['times'] = times
    for i in range(new_pos.shape[1]):
        data_out['y_pos{}'.format(i)] = new_pos[:,i]
    df = pd.DataFrame(data_out)
    df.to_csv('{}.csv'.format('.'.join(filepath.split('.')[:-1])))

