import ctypes
import h5py
from multiprocessing import cpu_count, RawArray, Pool, Manager
import numpy as np
import time

import gv

def run(segment_length, median_range):
    print('Run Median subtraction + Range normalization')

    ### Create dataset if necessary
    gv.f.require_dataset(gv.KEY_PROCESSED,
                         shape=gv.dset.shape,
                         dtype=np.uint8,
                         chunks=(1, *gv.f[gv.KEY_ORIGINAL].shape[1:]))
    t,x,y,c = gv.dset.shape

    ### Start timing
    tstart = time.perf_counter()

    ### Close file so subprocesses can open (r) it safely
    dset_name = gv.dset.name
    gv.f.close()

    ### Create output array
    c_video_out = RawArray(ctypes.c_uint8, t*x*y*c)
    video_out = np.frombuffer(c_video_out, dtype=np.uint8).reshape((t,x,y,c))
    video_out[:,:,:] = 0

    ### Progress list
    manager = Manager()
    progress = manager.list()

    segments = np.arange(0, t, segment_length, dtype=int)
    print('Video frames', t)
    print('Video segments', segments)
    gv.statusbar.startProgress('Median subtraction + Range Normalization...', len(segments) * 2 - 1)

    process_num = cpu_count()-2
    print('Using {} subprocesses'.format(process_num))
    with Pool(process_num, initializer=init_worker, initargs=(c_video_out, dset_name, progress, segment_length, median_range, gv.filepath)) as p:

        print('Calculate medians')
        r1 = p.map_async(worker_calc_pixel_median, segments)
        while not(r1.ready()):
            time.sleep(1/10)
            gv.statusbar.setProgress(len(progress))


    gv.statusbar.endProgress()

    gv.f = h5py.File(gv.filepath, 'a')

    print('Time for execution:', time.perf_counter()-tstart)

    ### Save to file
    gv.statusbar.startBlocking('Saving...')
    gv.f[gv.KEY_PROCESSED][:] = video_out
    gv.statusbar.setReady()
    gv.w.setDataset(gv.KEY_PROCESSED)
    gv.w.gb_med_norm.setChecked(False)

################
## Worker functions

# Globals
c_video_out = None
c_minmax_val = None
data_shape = None
array_out = None
index_list = None
filepath = None
f = None
dset_name = None
median_range = None
segment_length = None

def init_worker(cvidout, dsetn, idx_list, slength, mrange, fpath):
    global c_video_out, array_out, dset_name, data_shape, index_list, median_range, segment_length, filepath, f
    median_range = mrange
    filepath = fpath
    f = h5py.File(filepath, 'r')
    c_video_out = cvidout
    dset_name = dsetn
    data_shape = f[dset_name].shape
    array_out = np.frombuffer(c_video_out, dtype=np.uint8).reshape(data_shape)
    index_list = idx_list
    segment_length = slength

def worker_calc_pixel_median(start_idx):
    global array_out, data_shape, dset_name, index_list, segment_length, median_range, f
    end_idx = start_idx + segment_length
    print('Slice {} to {}'.format(start_idx, end_idx))

    index_list.append(('median_started', start_idx, end_idx))


    medrange = 20
    out = np.empty((segment_length, *data_shape[1:]))
    for i in range(start_idx, end_idx):

        start = i - medrange
        end = i + medrange
        if start < 0:
            end -= start
            start = 0

        if end > data_shape[0]:
            start -= end - data_shape[0]
            end = data_shape[0]

        out[i-start_idx,:,:,:] = f[dset_name][i,:,:,:] - np.median(f[dset_name][start:end,:,:,:], axis=0)
        #array_out[i, :, :, :] = median_norm(f[dset_name][i, :, :, :], f[dset_name][start:end, :, :, :])

    out -= out.min()
    out /= out.max()
    array_out[start_idx:end_idx,:,:,:] = (out * (2**8-1)).astype(np.uint8)

    print('Slice {} to {} finished'.format(start_idx, end_idx, 'finished'))

    index_list.append(('median_finished', start_idx, end_idx))
    return start_idx

def median_norm(frame, slice):
    out = frame - np.median(slice, axis=0).astype(np.float32)
    out -= out.min()
    out /= out.max()
    return (out * (2**8-1)).astype(np.uint8)