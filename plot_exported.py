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

#filepath = 'test_pickle/swimheight_light_5cm_Scale50pc.pickle' # Very good
#filepath = 'test_pickle/swimheight_dark_5cm_Scale100pc.pickle' # Good
filepath = 'test_pickle/swimheight_light_10cm_Scale100pc.pickle' # Okay
#filepath = 'test_pickle/swimheight_dark_15cm_Scale100pc.pickle' # Terrible problems
with open(filepath, 'rb') as file:
    data = pickle.load(file)

positions = data['positions']
times = data['time']

### Write to uniform array
pos = np.nan * np.ones((len(positions), 50, 2))
for i, c in enumerate(positions):
    for j, cc in enumerate(c):
        pos[i,j,:] = cc

times = times[5:3000]
pos = pos[5:3000,:,:]

colors = ['b', 'r', 'g', 'c', 'y', 'm', 'k']
plt.figure()

### Plot first few particles
plot_first = 0
plot_num = 6
data_out = dict()
data_out['times'] = times


def filter_position(p, fps, cutoff, discard_factor, restitute_factor, plot=False):

    if np.isnan(pos[:,i,1]).sum() > 0:
        print('WARNING: {} NaNs in position dataset. '.format(np.isnan(pos[:,i,1]).sum())
              + 'Substituting with zeros for filtering')
        p[np.isnan(p)] = 0.

    p_orig = np.copy(p)

    ### Plot before
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

    plt.plot(times, p, '-', color=colors[i], linewidth=1.2, alpha=1.0, label='Fish {}'.format(i))

    return p


for i in range(plot_first, plot_first+plot_num):
    print('Fish {}'.format(i))


    p = filter_position(p=pos[:, i, 1],
                        fps=5,
                        cutoff=0.1,
                        discard_factor=0.5,
                        restitute_factor=5.0,
                        plot=True)


    ### (optional) Inspect speed histogram
    if False:
        n, bins, _ = plt.hist(local_speeds, bins=100, density=True, color=colors[i], alpha=0.1)
        bin_centers = bins + (bins[1]-bins[0])/2
        print(stats.norm.logpdf(bin_centers, loc=speed_m, scale=speed_sd).min() > 0.)
        plt.plot(bin_centers, stats.norm.pdf(bin_centers, loc=speed_m, scale=speed_sd), color=colors[i])
        plt.axvline(speed_m-2*speed_sd, 0, 1, color=colors[i])
        plt.axvline(speed_m+2*speed_sd, 0, 1, color=colors[i])

        plt.yscale('log')
        plt.show()

    data_out['y_pos{}'.format(i)] = p

plt.ylabel('Y-Position [cm]')
plt.xlabel('Time [s]')
plt.show()

df = pd.DataFrame(data_out)
df.to_csv('{}.csv'.format('.'.join(filepath.split('.')[:-1])))

