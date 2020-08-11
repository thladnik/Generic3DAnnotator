

import numpy as np
import matplotlib.pyplot as plt
import pickle

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
for i in range(plot_first):
    boolvec = (pos[:,i,1] >= 0.0) & (pos[:,i,1] < 6.0)
    t = times[boolvec]
    p = pos[boolvec,i,1]
    plt.plot(t, p, '--', alpha=0.4, label='Fish {}'.format(i))

plt.legend()
plt.ylabel('Y-Position [cm]')
plt.xlabel('Time [s]')
plt.show()
