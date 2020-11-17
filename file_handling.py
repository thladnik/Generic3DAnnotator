"""
https://github.com/thladnik/Generic3DAnnotator/file_handling.py - Importing raw data and opening imported hdf5 files.
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
import h5py
import numpy as np
import os
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QLabel
import time
from trackpy import PandasHDFStoreBig

import gv

################################################################
# File handling functions


def open_file():

    # Query file path
    fileinfo = QtWidgets.QFileDialog.getOpenFileName(gv.w, 'Open file...', gv.open_dir,
                                                     f'Annotation file (*.{gv.EXT_ANNOT});;')

    if fileinfo == ('', ''):
        return

    # First close any open file
    close_file()

    # Set filepath
    gv.filepath = fileinfo[0]
    print('Open file {}'.format(gv.filepath))

    # Open files
    gv.h5f = h5py.File(gv.filepath, 'a')

    phases = [s for s in gv.h5f.keys() if s.startswith('phase')]
    # No phase entries?
    if not(bool(phases)):
        gv.h5f.close()
        print(f'File {gv.filepath} empty or not compatible')
        return

    # Select first phase
    gv.h5f = gv.h5f[phases[0]]

    open_trackpy()

    # Set video
    gv.w.set_title(gv.filepath)
    gv.w.update_ui()

    gv.w.rpanel.setEnabled(True)

def get_trackpy_path():
    if gv.h5f.name == '/':
        filepath = f'{gv.filepath}'.replace(f'.{gv.EXT_ANNOT}', f'.{gv.EXT_TRACKPY}')
    else:
        filepath = f'{gv.filepath}_{gv.h5f.name.replace("/", "")}'.replace(f'.{gv.EXT_ANNOT}', f'.{gv.EXT_TRACKPY}')

    return filepath

def open_trackpy():

    if gv.filepath is None:
        return

    # Close old one
    if gv.tpf is not None:
        gv.tpf.close()
        gv.tpf = None

    # Look for new one
    filepath = get_trackpy_path()
    if not(os.path.exists(filepath)):
        print('No trackpy container found.')
        return

    # Open new one
    print(f'Open trackpy container {filepath}')
    gv.tpf = PandasHDFStoreBig(filepath, 'a')


def close_file():
    if gv.h5f is not None:
        print(f'Close file {gv.h5f.file.filename}')
        gv.h5f.file.close()
        gv.h5f = None

    if gv.tpf is not None:
        print(f'Close file {gv.tpf.filename}')
        gv.tpf.close()
        gv.tpf = None


    gv.filepath = None
    gv.w.set_title()
    gv.w.rpanel.setEnabled(False)
    gv.w.viewer.update_image()
    gv.w.viewer.update_roi_display()

################################################################
# Video import functions

def import_file():
    """Import a new video/image sequence-type file and create mem-mapped file
    """
    # TODO: add rotate ON import
    #       do NOT do rotations AFTER importing, it's not worth the trouble

    fileinfo = QtWidgets.QFileDialog.getOpenFileName(gv.w, 'Open file...', gv.open_dir,
                                                     '[HDF] Container (*.h5; *.hdf5);;'
                                                     '[Monochrome] Video Files (*.avi; *.mp4);;'
                                                     '[RGB] Video Files (*.avi; *.mp4);;'
                                                     )

    if fileinfo == ('', ''):
        return

    close_file()

    videopath = fileinfo[0]
    videoformat = fileinfo[1]

    ### Get video file info
    path_parts = videopath.split(os.path.sep)
    gv.open_dir = os.path.join(videopath[:-1])
    filename = path_parts[-1]
    ext = filename.split('.')[-1]

    ### Set file path and handle
    gv.filepath = os.path.join(gv.open_dir, f'{filename[:-(len(ext) + 1)]}.{gv.EXT_ANNOT}')
    print('Import file {} to {}'.format(videopath, gv.filepath))
    if os.path.exists(gv.filepath):
        confirm_dialog = QtWidgets.QMessageBox.question(gv.w, 'Overwrite file?',
                                                        'This file has already been imported. Do you want to re-import and overwrite?',
                                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                                                        QtWidgets.QMessageBox.No)

        if confirm_dialog == QtWidgets.QMessageBox.No:
            open_file()
            return
        elif confirm_dialog == QtWidgets.QMessageBox.Cancel:
            return

    # Open file
    gv.h5f = h5py.File(gv.filepath, 'w')

    # IMPORT
    props = dict()
    if ext.lower() in ['avi']:
        props = import_avi(videopath, videoformat)
    elif ext.lower() in ['mp4']:
        pass
    elif ext.lower() in ['h5', 'hdf5']:
        props = import_hdf5(videopath, videoformat)
    else:
        close_file()
        return

    if not(bool(props)):
        # Add ATTRIBUTES
        dialog = QtWidgets.QDialog(gv.w)
        dialog.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)
        dialog.setLayout(QtWidgets.QGridLayout())
        dialog.lbl_set = QLabel('Set metadata')
        dialog.lbl_set.setStyleSheet('font-weight:bold; text-alignment:center;')
        dialog.layout().addWidget(dialog.lbl_set, 0, 0, 1, 2)

        dialog.fields = dict()
        for i, (key, val) in enumerate(props.items()):
            dialog.layout().addWidget(QLabel(key), i + 1, 0)
            if isinstance(val, (float, np.float32, np.float64)):
                field = QtWidgets.QDoubleSpinBox()
                field.setValue(val)
            elif isinstance(val, (int, np.uint)):
                field = QtWidgets.QSpinBox()
                field.setValue(val)
            else:
                field = QtWidgets.QLineEdit()
                field.setText(val)

            dialog.fields[key] = field
            dialog.layout().addWidget(field, i + 1, 1)

        dialog.btn_submit = QtWidgets.QPushButton('Save')
        dialog.btn_submit.clicked.connect(dialog.accept)
        dialog.layout().addWidget(dialog.btn_submit, len(props) + 2, 0, 1, 2)

        if not (dialog.exec_() == QtWidgets.QDialog.Accepted):
            raise Exception('No scale for limits set.')

        for key, field in dialog.fields.items():
            if hasattr(field, 'value'):
                gv.h5f.attrs[key] = field.value()
            elif hasattr(field, 'text'):
                gv.h5f.attrs[key] = field.text()

    # Set indices and timepoints
    if gv.KEY_FRAME_IDCS not in gv.h5f:
        gv.h5f.create_dataset(gv.KEY_FRAME_IDCS, data=np.arange(gv.h5f[gv.KEY_ORIGINAL].shape[0]), dtype=np.uint64)
    if gv.KEY_TIME not in gv.h5f:
        gv.h5f.create_dataset(gv.KEY_TIME, data=gv.h5f[gv.KEY_FRAME_IDCS], dtype=np.float64)
        if gv.KEY_ATTR_FPS in gv.h5f.attrs:
            gv.h5f[gv.KEY_TIME][:] = gv.h5f[gv.KEY_TIME][:] / gv.h5f.attrs[gv.KEY_ATTR_FPS]

    # Set video
    gv.w.set_title(gv.filepath)
    gv.w.viewer.slider.setValue(0)
    #gv.w.set_dataset(gv.KEY_ORIGINAL)

    gv.w.rpanel.setEnabled(True)


################################
# HDF5

def import_hdf5(hdf5path, format):
    source = h5py.File(hdf5path, 'r')

    phases = [s for s in source.file.keys() if s.startswith('phase')]

    if not(bool(phases)):
        print('No phases in HDF5 file. Abort import.')
        return

    group = source[phases[0]]
    avail_datasets = [f'{grp_name}/{dset_name}' for grp_name, grp in group.items() for dset_name, dset in grp.items() ]

    # Let user select dataset names for frame time and data
    dialog = QtWidgets.QDialog(gv.w)
    dialog.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)
    dialog.setLayout(QtWidgets.QGridLayout())
    dialog.lbl_set = QLabel('Select datasets')
    dialog.lbl_set.setStyleSheet('font-weight:bold; text-alignment:center;')
    dialog.layout().addWidget(dialog.lbl_set, 0, 0, 1, 2)
    # Frame dataset name
    dialog.layout().addWidget(QLabel('Frames'), 1, 0)
    dialog.cb_frame_name = QtWidgets.QComboBox()
    dialog.cb_frame_name.addItems(avail_datasets)
    dialog.layout().addWidget(dialog.cb_frame_name, 1, 1)
    # Frame time name
    dialog.layout().addWidget(QLabel('Times'), 2, 0)
    dialog.cb_time_name = QtWidgets.QComboBox()
    dialog.cb_time_name.addItems(avail_datasets)
    dialog.layout().addWidget(dialog.cb_time_name, 2, 1)

    dialog.btn_submit = QtWidgets.QPushButton('Confirm')
    dialog.btn_submit.clicked.connect(dialog.accept)
    dialog.layout().addWidget(dialog.btn_submit, 3, 0, 1, 2)

    if not (dialog.exec_() == QtWidgets.QDialog.Accepted):
        raise Exception('Need to set frame dataset and time dataset names')

    frame_name = dialog.cb_frame_name.currentText()
    time_name = dialog.cb_time_name.currentText()

    if frame_name == time_name:
        print('ERROR: Frame time and frame data cannot be the same')
        return

    # Rotate/mirror
    dialog = QtWidgets.QDialog(gv.w)
    dialog.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)
    dialog.setLayout(QtWidgets.QGridLayout())
    # Set image view
    from gui import HDF5ImageView
    dialog.imview = HDF5ImageView()
    # Set rotation/flip
    dialog.cb_rotation = QtWidgets.QComboBox()
    dialog.cb_rotation.addItems(['None', '90CCW', '180', '270CCW'])
    dialog.cb_rotation.currentIndexChanged.connect(lambda i: dialog.imview.set_rotation(i))
    dialog.cb_rotation.currentIndexChanged.connect(dialog.imview.update_image)
    dialog.layout().addWidget(dialog.cb_rotation, 0, 0)
    dialog.check_flip_ud = QtWidgets.QCheckBox('Flip vertical')
    dialog.check_flip_ud.stateChanged.connect(lambda: dialog.imview.set_flip_ud(dialog.check_flip_ud.isChecked()))
    dialog.check_flip_ud.stateChanged.connect(dialog.imview.update_image)
    dialog.layout().addWidget(dialog.check_flip_ud, 0, 1)
    dialog.check_flip_lr = QtWidgets.QCheckBox('Flip horizontal')
    dialog.check_flip_lr.stateChanged.connect(lambda: dialog.imview.set_flip_lr(dialog.check_flip_lr.isChecked()))
    dialog.check_flip_lr.stateChanged.connect(dialog.imview.update_image)
    dialog.layout().addWidget(dialog.check_flip_lr, 0, 2)
    #dialog.imview.set_dataset(group[frame_name])
    dialog.imview.setFixedSize(*[max(group[frame_name].shape)] * 2)
    dialog.layout().addWidget(dialog.imview, 1, 0)
    dialog.btn_submit = QtWidgets.QPushButton('Confirm')
    dialog.btn_submit.clicked.connect(dialog.accept)
    dialog.layout().addWidget(dialog.btn_submit, 2, 0)

    if not (dialog.exec_() == QtWidgets.QDialog.Accepted):
        raise Exception('Need to confirm rotation/flip')

    # Import
    for j, phase_name in enumerate(phases):
        # Get time dimension size
        t_dim = source[phase_name][time_name].shape[0]

        gv.statusbar.start_progress(f'Import phase {j+1} of {len(phases)} from {hdf5path}', t_dim)

        # Create group
        gv.h5f = gv.h5f.file.create_group(phase_name)

        # Set time
        gv.h5f.create_dataset(gv.KEY_TIME, data=source[phase_name][time_name])

        # Set frames
        for i, im in enumerate(source[phase_name][frame_name]):
            dset = gv.h5f.require_dataset(gv.KEY_ORIGINAL,
                                          shape=(t_dim, *im.shape),
                                          dtype=np.uint8,
                                          chunks=(1, *im.shape),
                                          compression='gzip',
                                          compression_opts=9)

            dset[i] = dialog.imview._rotate(dialog.imview._flip(im))

            gv.statusbar.set_progress(i)

    gv.statusbar.end_progress()

    return dict()


################################
# AVI

def import_avi(videopath, format):
    import av

    mono = False
    if format.startswith('[Monochrome]'):
        mono = True

    tstart = time.time()


    for k in range(3):
        # Open video file
        vhandle = av.open(videopath)
        v = vhandle.streams.video[0]

        props = {
            gv.KEY_ATTR_FPS: int(round(v.base_rate.numerator / v.base_rate.denominator)),
        }

        # Create group for single-file import
        gv.h5f = gv.h5f.file.create_group(f'phase_{k}')

        # Number of frames
        t_dim = v.frames

        gv.statusbar.start_progress('Importing {}'.format(videopath), t_dim)
        for i, image in enumerate(vhandle.decode()):

            # Get image ndarray
            im = np.asarray(image.to_image()).astype(np.uint8)

            # Convert to monochrome if necessary
            if mono:
                im = im[:, :, 0][:, :, np.newaxis]

            dset = gv.h5f.require_dataset(gv.KEY_ORIGINAL,
                                   shape=(t_dim, *im.shape),
                                   dtype=np.uint8,
                                   chunks=(1, *im.shape),
                                   compression=None)

            # Update progress
            if i % 100 == 0:
                gv.statusbar.set_progress(i)

            # Set frame data
            # gv.h5f[gv.KEY_ORIGINAL][i, :, :, :] = im[:, :, :]
            dset[i, :, :, :] = im[:, :, :]

        print('Import finished after {:.2f} seconds'.format(time.time() - tstart))

    gv.statusbar.end_progress()

    return props