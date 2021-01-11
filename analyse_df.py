import pandas as pd
import os
from export_data import base_folder
import matplotlib.pyplot as plt


def calc_x_velocity(df):
    # Calculate frame-by-frame time and x position differences
    dt = df.time.diff()
    dx = df.x.diff()

    # Calculate velocity form position delta and time delta
    vx = dx/dt

    # Visualization (does it for each particle (MANY particles in Df) -> not for analysis
    # plt.plot(df.time.values - df.time.values[0], vx)
    # plt.show()

    # Inspect:
    # import IPython
    # IPython.embed()

    # Return mean x velocity for this particular particle
    return vx.mean()

def calc_y_velocity(df):
    # Calculate frame-by-frame time and y position differences
    dt = df.time.diff()
    dy = df.y.diff()

    # Calculate velocity form position delta and time delta
    vy = dy/dt

    # Visualization (does it for each particle (MANY particles in Df) -> not for analysis
    # plt.plot(df.time.values - df.time.values[0], vy)
    # plt.show()

    # Inspect:
    # import IPython
    # IPython.embed()

    # Return mean y velocity for this particular particle
    return vy.mean()


if __name__ == '__main__':
    Df = pd.read_hdf(os.path.join(base_folder,'Summary.h5'),'all')

    grps = Df.groupby(['folder', 'phase_name', 'particle'])

    grp_df = pd.DataFrame()
    grp_df['x_vel'] = grps.apply(calc_x_velocity)
    grp_df['y_vel'] = grps.apply(calc_y_velocity)

    import IPython
    IPython.embed()