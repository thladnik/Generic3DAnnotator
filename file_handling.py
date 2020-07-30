import h5py
import numpy as np
import os
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QLabel
import time

import gv
import object_handling

################################################################
### File handling functions

def open_file():

    ### Query file path
    fileinfo = QtWidgets.QFileDialog.getOpenFileName(gv.w, 'Open file...', gv.open_dir,
                                                     'Imported HDF5 (*.hdf5);;')

    if fileinfo == ('', ''):
        return

    ### First close any open file
    close_file()

    ### Set filepath
    gv.filepath = fileinfo[0]
    print('Open file {}'.format(gv.filepath))

    ### Open file
    gv.f = h5py.File(gv.filepath, 'a')

    ### Set video
    gv.w.setDataset(gv.KEY_ORIGINAL)
    gv.w.updateRotationFilter(0)
    # main.update_axes_table()

    gv.w.setTitle(gv.filepath)

    gv.w.rpanel.setEnabled(True)

    #gv.w.updateParticleMarkers()


def close_file():
    if not(gv.f is None):
        print('Close file {}'.format(gv.f.filename))
        gv.f.close()
        gv.f = None


    object_handling.clear_objects()

    gv.filepath = None
    gv.w.setTitle()
    gv.w.setDataset(None)
    gv.w.rpanel.setEnabled(False)


################################################################
### Video import functions

def import_file():
    """Import a new video/image sequence-type file and create mem-mapped file
    """

    fileinfo = QtWidgets.QFileDialog.getOpenFileName(gv.w, 'Open file...', gv.open_dir,
                                                     '[Monochrome] Video Files (*.avi; *.mp4);;'
                                                     '[RGB] Video Files (*.avi; *.mp4)')

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
    gv.filepath = os.path.join(gv.open_dir, '{}.hdf5'.format(filename[:-(len(ext) + 1)]))
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

    ### Open file
    gv.f = h5py.File(gv.filepath, 'w')

    ################
    ### IMPORT

    ### (Include file format options here)
    props = dict()
    if ext.lower() == 'avi':
        props = import_avi(videopath, videoformat)
    elif ext.lower() == 'mp4':
        pass
    else:
        close_file()
        return

    ################
    ### Add ATTRIBUTES
    dialog = QtWidgets.QDialog(gv.w)
    dialog.setWindowTitle('Set metadata')
    dialog.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)
    dialog.setLayout(QtWidgets.QGridLayout())
    dialog.lbl_set = QLabel('Set scale')
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
            gv.f.attrs[key] = field.value()
        elif hasattr(field, 'text'):
            gv.f.attrs[key] = field.text()

    ### Set indices and timepoints
    gv.f.create_dataset(gv.KEY_FRAMEIDCS, data=np.arange(gv.f[gv.KEY_ORIGINAL].shape[0]), dtype=np.uint64)
    gv.f.create_dataset(gv.KEY_TIME, data=gv.f[gv.KEY_FRAMEIDCS], dtype=np.float64)
    if gv.KEY_FPS in gv.f.attrs:
        gv.f[gv.KEY_TIME][:] = gv.f[gv.KEY_TIME][:] / gv.f.attrs[gv.KEY_FPS]

    ### Set video
    gv.w.setTitle(gv.filepath)
    gv.w.setDataset(gv.KEY_ORIGINAL)

    gv.w.rpanel.setEnabled(True)


################################
### AVI

def import_avi(videopath, format):
    import av

    mono = False
    if format.startswith('[Monochrome]'):
        mono = True

    tstart = time.time()
    ### Open video file
    vhandle = av.open(videopath)
    v = vhandle.streams.video[0]

    props = {
        gv.KEY_FPS: int(round(v.base_rate.numerator / v.base_rate.denominator)),
    }

    ### Import frames
    t_dim = v.frames

    gv.statusbar.startProgress('Importing {}'.format(videopath), t_dim)

    for i, image in enumerate(vhandle.decode()):

        ### Get image ndarray
        im = np.asarray(image.to_image()).astype(np.uint8)

        ### Convert to monochrome if necessary
        if mono:
            im = im[:, :, 0][:, :, np.newaxis]

        gv.f.require_dataset(gv.KEY_ORIGINAL,
                             shape=(t_dim, *im.shape),
                             dtype=np.uint8,
                             chunks=(1, *im.shape),
                             compression=None)

        ### Update progress
        gv.statusbar.setProgress(i)
        gv.app.processEvents()
        ### Set frame data
        gv.f[gv.KEY_ORIGINAL][i, :, :, :] = im[:, :, :]

    print('Import finished after {:.2f} seconds'.format(time.time() - tstart))

    gv.statusbar.endProgress()

    return props