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

filepath = 'T:/swimheight_light_5cm_Scale50pc.pickle'
with open(filepath, 'rb') as file:
    data = pickle.load(file)

positions = data['positions']
times = data['time']

### Write to uniform array
pos = np.nan * np.ones((len(positions), 50, 2))
for i, c in enumerate(positions):
    for j, cc in enumerate(c):
        pos[i,j,:] = cc

### Plot first few particles
plot_first = 9
data_out = dict()
data_out['times'] = times

for i in range(plot_first):
    boolvec = (pos[:,i,1] >= 0.0) & (pos[:,i,1] < 6.0)
    p = pos[:,i,1]
    p[np.logical_not(boolvec)] = np.nan
    data_out['y_pos{}'.format(i)] = p
    plt.plot(times, p, '--', alpha=0.4, label='Fish {}'.format(i))

plt.legend()
plt.ylabel('Y-Position [cm]')
plt.xlabel('Time [s]')
plt.show()

df = pd.DataFrame(data_out)
df.to_csv('{}.csv'.format('.'.join(filepath.split('.')[:-1])))

