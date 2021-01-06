import os
from gv import  EXT_TRACKPY, KEY_ATTR_ROI_XLEN, KEY_ATTR_ROI_YLEN
import h5py
import trackpy as tp
import pandas as pd
import numpy as np
from collections.abc import Iterable

display_parameter_file = 'Display.hdf5'
gaf_file = 'Camera.gaf'

base_folder = './data'

def read_store(filepath):
    """Read and return all per-frame entries from trackpy file"""

    f = tp.PandasHDFStoreBig(filepath, mode='r')
    dfs = list()
    for idx in f.frames:
        dfs.append(f.get(idx))

    f.close()

    return pd.concat(dfs)

def read_folder_contents(folder_path):
    # If no annotation file exists, return none
    if not(os.path.exists(os.path.join(folder_path, gaf_file))):
        print('Empty')
        return None

    # Open annotation file
    gaf_h5 = h5py.File(os.path.join(folder_path, gaf_file), 'r')

    # Open display file
    display_h5 = h5py.File(os.path.join(folder_path, display_parameter_file), 'r')

    # Collect all trackpy files
    files = os.listdir(folder_path)
    files = [s for s in files if not(os.path.isdir(os.path.join(folder_path, s)))]
    files = [s for s in files if s[-len(EXT_TRACKPY):] == EXT_TRACKPY]
    print(f'Read phase files {files}')
    dfs = list()
    for filename in files:
        # Remove extension from filename and split to get phase_id
        phase_name = '_'.join(filename.replace(f'.{EXT_TRACKPY}', '').split('_')[1:3])
        phase_id = int(filename.replace(f'.{EXT_TRACKPY}', '').split('_')[2])

        print(f'Read {filename}')

        try:

            # Read tracking results and add stimulation phase information
            df = read_store(os.path.join(folder_path, filename))
            df['folder'] = folder_path
            df['file'] = filename
            df['phase_id'] = phase_id
            df['phase_name'] = phase_name

            # Scale position data
            gaf_grp = gaf_h5[phase_name]
            xscale = gaf_grp.attrs[KEY_ATTR_ROI_XLEN]
            yscale = gaf_grp.attrs[KEY_ATTR_ROI_YLEN]
            df['x'] *= xscale
            df['x'] -= xscale/2
            df['y'] *= yscale
            df['y'] -= yscale/2

            display_grp = display_h5[phase_name]

            # Write display attributes to Df
            for k, v in display_grp.attrs.items():

                if isinstance(v, Iterable):
                    # If it's an iterable, add one column per item
                    for i, vv in enumerate(v):
                        df[f'{k}_{i}'] = vv
                else:
                    df[k] = v

            # Add to list
            dfs.append(df)

        except Exception as exc:
            print(f'Failed processing {filename} // Exception: {exc}')

    # Close files
    gaf_h5.close()
    display_h5.close()

    if len(dfs) > 0:
        return pd.concat(dfs)
    else:
        return None


if __name__ == '__main__':

    dfs = list()
    for s in os.listdir(base_folder):
        path = os.path.join(base_folder, s)
        if not(os.path.isdir(path)):
            continue

        print(f'Read folder {path}')
        df = read_folder_contents(path)

        if df is not None:
            dfs.append(df)

    # Concat all
    Df = pd.concat(dfs)
    Df.to_hdf(os.path.join(base_folder, 'Summary.h5'), 'all')

    import IPython
    IPython.embed()
