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

