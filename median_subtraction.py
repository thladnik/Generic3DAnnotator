"""
https://github.com/thladnik/Generic3DAnnotator/median_subtraction.py - Sliding median subtraction script + normalization.
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
import ctypes
import h5py
from multiprocessing import cpu_count, RawArray, Pool, Manager
import numpy as np
import time

import gv

def run(segment_length, median_range, coords):
    """

    var coords: array with shape (2,x_dim,y_dim),
                 where coords[0,:,:] contains the x
                 and coords[1,:,:] contains the y indices
    """

    group_name = gv.h5f.name

    slice_x = coords[0,:,:]
    slice_y = coords[1,:,:]

    t_dim, x_dim, y_dim = (gv.h5f[gv.KEY_ORIGINAL].shape[0],
                                  coords.shape[1],
                                  coords.shape[2])


    print('Run Median subtraction + Range normalization')
    # Create dataset (delete previous if necessary
    if gv.KEY_PROCESSED in gv.h5f:
        del gv.h5f[gv.KEY_PROCESSED]

    gv.h5f.create_dataset(gv.KEY_PROCESSED,
                          shape=(t_dim, x_dim, y_dim),
                          dtype=np.uint8,
                          chunks=(1, x_dim, y_dim))

    # Split up into shorter segments for workers
    segments = np.arange(0, t_dim, segment_length, dtype=int)

    # Print debug info
    print(f'Video frames {t_dim}')
    print(f'Video segments{segments}')
    print(f'Median range {median_range}')
    print(f'In shape {gv.h5f[gv.KEY_ORIGINAL].shape}')
    print(f'Out shape {gv.h5f[gv.KEY_PROCESSED].shape}')
    gv.statusbar.start_progress('Median subtraction + Range Normalization...', len(segments) * 2 - 1)
    gv.statusbar.set_progress(0)

    # Start timing
    tstart = time.perf_counter()

    # Close file so subprocesses can open (r) it safely
    dset_name = gv.KEY_ORIGINAL
    filepath = gv.filepath
    gv.h5f.file.close()

    # Create output array
    c_video_out = RawArray(ctypes.c_uint8, t_dim*x_dim*y_dim)
    video_out = np.frombuffer(c_video_out, dtype=np.uint8).reshape((t_dim, x_dim, y_dim))
    video_out[:,:,:] = 0

    # Progress list
    manager = Manager()
    progress = manager.list()


    initargs_ = (c_video_out,
                 (t_dim, x_dim, y_dim),
                  slice_x,
                  slice_y,
                  dset_name,
                  progress,
                  segment_length,
                  median_range,
                  filepath,
                  group_name)

    process_num = cpu_count()-2
    if process_num > len(segments):
        process_num = len(segments)
    print('Using {} subprocesses'.format(process_num))
    with Pool(process_num, initializer=init_worker, initargs=initargs_) as p:

        print('Calculate medians')
        r1 = p.map_async(worker_calc_pixel_median, segments)
        while not(r1.ready()):
            time.sleep(1/10)
            gv.statusbar.set_progress(len(progress))

    gv.statusbar.end_progress()

    # Open file again and select group
    gv.h5f = h5py.File(gv.filepath, 'a')
    gv.h5f = gv.h5f[group_name]

    print('Time for execution:', time.perf_counter()-tstart)

    # Save to file
    gv.statusbar.start_blocking('Saving...')
    gv.h5f[gv.KEY_PROCESSED][:] = video_out
    gv.statusbar.set_ready()


################
## Worker functions

# Globals
c_video_out = None
video_out_shape = None
slice_x = None
slice_y = None
dset_name = None
index_list = None
segment_length = None
median_range = None
filepath = None
group_name = None

f = None
array_out = None
sub_array_out = None

def init_worker(_c_video_out,
                _video_out_shape,
                _slice_x,
                _slice_y,
                _dset_name,
                _index_list,
                _segment_length,
                _median_range,
                _filepath,
                _group_name):
    global c_video_out, video_out_shape, slice_x, slice_y, dset_name, index_list, segment_length, median_range, filepath, \
        f, array_out, sub_array_out, gorup_name

    c_video_out = _c_video_out
    video_out_shape = _video_out_shape
    slice_x = _slice_x
    slice_y = _slice_y
    dset_name = _dset_name
    index_list = _index_list
    segment_length = _segment_length
    median_range = _median_range
    filepath = _filepath
    group_name = _group_name

    # Open file
    f = h5py.File(filepath, 'r')
    f = f[group_name]
    # Set subset array
    sub_array_out = np.empty((segment_length, *video_out_shape[1:]))
    # Set shared out array
    array_out = np.frombuffer(c_video_out, dtype=np.uint8).reshape(video_out_shape)

def worker_calc_pixel_median(start_idx):
    global array_out, sub_array_out, video_out_shape, dset_name, index_list, segment_length, median_range, f, slice_x, slice_y
    end_idx = start_idx + segment_length
    print('Slice {} to {}'.format(start_idx, end_idx))

    index_list.append(('median_started', start_idx, end_idx))

    try:
        stop_idx = segment_length
        # Calculate running median for each pixel in frame
        for i in range(start_idx, end_idx):
            if i >= array_out.shape[0]:
                stop_idx = i % segment_length
                print(f'In seg {start_idx}', f'Break at {i}')
                break

            start = i - median_range
            end = i + median_range
            if start < 0:
                end -= start
                start = 0

            if end > video_out_shape[0]:
                start -= end - video_out_shape[0]
                end = video_out_shape[0]


            frame = f[dset_name][i, :, :]
            sub_frame = frame[slice_x, slice_y]

            frames = f[dset_name][start:end, :, :]
            sub_frames = frames[:, slice_x, slice_y]

            sub_array_out[i-start_idx, :, :] = sub_frame - np.median(sub_frames, axis=0)

        sub_array_out[sub_array_out[:,:,:] < 0] = 0
        sub_array_out[sub_array_out[:,:,:] > 255] = 255

        # Write to out
        array_out[start_idx:start_idx+stop_idx, :, :] = sub_array_out[:stop_idx,:,:].astype(np.uint8)
        # Normalize
        # sub_array_out -= sub_array_out.min()
        # sub_array_out /= sub_array_out.max()
        #
        # array_out[start_idx:end_idx,:,:] = (sub_array_out[:stop_idx,:,:] * (2**8-1)).astype(np.uint8)

    except Exception as exc:
        import traceback
        print(traceback.print_exc())

    print('Slice {} to {} finished'.format(start_idx, end_idx, 'finished'))

    index_list.append(('median_finished', start_idx, end_idx))
    return start_idx
