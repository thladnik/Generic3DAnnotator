"""
https://github.com/thladnik/Generic3DAnnotator/gui.py - Main file for starting program/GUI.
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
from matplotlib import cm
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QLabel
import pyqtgraph as pg
import time
import pickle
import trackpy as tp
import os
from queue import Queue

import gv
import file_handling
import median_subtraction

from io import StringIO
import sys

################################
### Main window

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        gv.w = self

        self.resize(1300, 1000)
        self.set_title()
        self.cw = QtWidgets.QWidget()
        self.setCentralWidget(self.cw)
        self.cw.setLayout(QtWidgets.QGridLayout())

        ################
        # Create menu

        self.mb = QtWidgets.QMenuBar()
        self.setMenuBar(self.mb)
        self.mb_file = self.mb.addMenu('File')
        self.mb_file_open = self.mb_file.addAction('Open file...')
        self.mb_file_open.setShortcut('Ctrl+O')
        self.mb_file_open.triggered.connect(file_handling.open_hdf5)
        self.mb_file_import = self.mb_file.addAction('Import image sequence...')
        self.mb_file_import.setShortcut('Ctrl+I')
        self.mb_file_import.triggered.connect(file_handling.import_file)
        self.mb_file_export = self.mb_file.addAction('Export scaled position data')
        self.mb_file_export.setShortcut('Ctrl+E')
        self.mb_file_export.triggered.connect(lambda: self.exportPositionData(scaled=True))
        self.mb_file_close = self.mb_file.addAction('Close file')
        self.mb_file_close.triggered.connect(file_handling.close_file)
        self.mb_edit = self.mb.addMenu('Edit')

        ################
        # Create ImageView
        self.viewer = HDF5ImageView(self, update_fun=self.update_image)
        self.cw.layout().addWidget(self.viewer, 0, 0)

        self.particle_markers = pg.ScatterPlotItem(size=10,
                                                   pen=pg.mkPen(None),
                                                   brush=pg.mkBrush(255, 0, 0, 255))
        self.particle_markers.sigClicked.connect(self.clicked_on_particle)
        self.viewer.view.addItem(self.particle_markers)

        ################
        # Create right panel

        self.rpanel = QtWidgets.QWidget()
        self.rpanel.setFixedWidth(350)
        self.rpanel.setLayout(QtWidgets.QVBoxLayout())
        self.cw.layout().addWidget(self.rpanel, 0, 1)

        ########
        # Display switch
        self.gb_display_switch = QtWidgets.QGroupBox('Switch display source')
        self.gb_display_switch.setLayout(QtWidgets.QVBoxLayout())
        self.gb_display_switch.btn_raw = QtWidgets.QPushButton('Raw')
        self.gb_display_switch.btn_raw.clicked.connect(lambda: self.set_dataset(gv.KEY_ORIGINAL))
        self.gb_display_switch.layout().addWidget(self.gb_display_switch.btn_raw)
        self.gb_display_switch.btn_processed = QtWidgets.QPushButton('Processed')
        self.gb_display_switch.btn_processed.clicked.connect(lambda: self.set_dataset(gv.KEY_PROCESSED))
        self.gb_display_switch.layout().addWidget(self.gb_display_switch.btn_processed)
        self.rpanel.layout().addWidget(self.gb_display_switch)

        label_pipeline = QtWidgets.QLabel('Processing pipeline')
        label_pipeline.setStyleSheet('font-weight:bold;')
        self.rpanel.layout().addWidget(label_pipeline)
        sep_line = QtWidgets.QFrame()
        sep_line.setFrameShape(QtWidgets.QFrame.HLine)
        sep_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.rpanel.layout().addWidget(sep_line)

        ########
        # 1. Import image sequence
        self.gb_import = QtWidgets.QGroupBox('1. Import image sequence')
        self.gb_import.setLayout(QtWidgets.QVBoxLayout())
        self.gb_import.btn_import = QtWidgets.QPushButton('Import...')
        self.gb_import.btn_import.clicked.connect(file_handling.import_file)
        self.gb_import.layout().addWidget(self.gb_import.btn_import)
        self.rpanel.layout().addWidget(self.gb_import)

        ########
        # Filter ROI contents
        self.gb_filter = QtWidgets.QGroupBox('2. Filter ROI contents')
        self.gb_filter.setLayout(QtWidgets.QGridLayout())
        # Segment length
        self.gb_filter.seg_len = QtWidgets.QSpinBox()
        self.gb_filter.seg_len.setMinimum(10)
        self.gb_filter.seg_len.setMaximum(9999)
        self.gb_filter.seg_len.setValue(200)
        self.gb_filter.layout().addWidget(QLabel('Seg. len. [lower = less RAM]'), 0, 0)
        self.gb_filter.layout().addWidget(self.gb_filter.seg_len, 0, 1)
        # Median range
        self.gb_filter.med_range = QtWidgets.QSpinBox()
        self.gb_filter.med_range.setMinimum(1)
        self.gb_filter.med_range.setMaximum(9999)
        self.gb_filter.med_range.setValue(50)
        self.gb_filter.layout().addWidget(QLabel('Range [lower = faster]'), 1, 0)
        self.gb_filter.layout().addWidget(self.gb_filter.med_range, 1, 1)
        # Run
        self.gb_filter.btn_run = QtWidgets.QPushButton('Run filter')
        self.gb_filter.btn_run.clicked.connect(self.run_roi_filter)
        self.gb_filter.layout().addWidget(self.gb_filter.btn_run, 2, 0, 1, 2)
        self.rpanel.layout().addWidget(self.gb_filter)

        ########
        # Particle detection
        self.gb_detection = QtWidgets.QGroupBox('3. Particle detection')
        self.gb_detection.setLayout(QtWidgets.QGridLayout())
        # Invert
        self.gb_detection.layout().addWidget(QtWidgets.QLabel('Black-on-white'), 0, 0)
        self.gb_detection.check_invert = QtWidgets.QCheckBox()
        self.gb_detection.layout().addWidget(self.gb_detection.check_invert, 0, 1)
        # Diameter
        self.gb_detection.layout().addWidget(QtWidgets.QLabel('Diameter'), 1, 0)
        self.gb_detection.spn_diameter = QtWidgets.QSpinBox()
        self.gb_detection.spn_diameter.valueChanged.connect(self.detection_diameter_validator)
        self.gb_detection.spn_diameter.setMinimum(1)
        self.gb_detection.spn_diameter.setMaximum(9999)
        self.gb_detection.spn_diameter.setValue(11)
        self.gb_detection.layout().addWidget(self.gb_detection.spn_diameter, 1, 1)
        # Minmass
        self.gb_detection.layout().addWidget(QtWidgets.QLabel('Min. brightness'), 2, 0)
        self.gb_detection.spn_minmass = QtWidgets.QSpinBox()
        self.gb_detection.spn_minmass.setMinimum(100)
        self.gb_detection.spn_minmass.setMaximum(9999)
        self.gb_detection.spn_minmass.setValue(100)
        self.gb_detection.layout().addWidget(self.gb_detection.spn_minmass, 2, 1)
        # Run
        self.gb_detection.btn_run = QtWidgets.QPushButton('Run')
        self.gb_detection.btn_run.clicked.connect(self.run_particle_detection)
        self.gb_detection.layout().addWidget(self.gb_detection.btn_run, 3, 0, 1, 2)
        self.rpanel.layout().addWidget(self.gb_detection)

        ########
        # Particle tracking
        self.gb_tracking = QtWidgets.QGroupBox('4. Trace particles')
        self.gb_tracking.setLayout(QtWidgets.QGridLayout())
        # Search range
        self.gb_tracking.layout().addWidget(QtWidgets.QLabel('Search range'), 0, 0)
        self.gb_tracking.srange = QtWidgets.QSpinBox()
        self.gb_tracking.srange.setMinimum(1)
        self.gb_tracking.srange.setMaximum(9999)
        self.gb_tracking.srange.setValue(10)
        self.gb_tracking.layout().addWidget(self.gb_tracking.srange, 0, 1)
        # Memory
        self.gb_tracking.layout().addWidget(QtWidgets.QLabel('Search memory'), 1, 0)
        self.gb_tracking.smemory = QtWidgets.QSpinBox()
        self.gb_tracking.smemory.setMinimum(1)
        self.gb_tracking.smemory.setMaximum(9999)
        self.gb_tracking.smemory.setValue(10)
        self.gb_tracking.layout().addWidget(self.gb_tracking.smemory, 1, 1)
        # Run
        self.gb_tracking.btn_run = QtWidgets.QPushButton('Run')
        self.gb_tracking.btn_run.clicked.connect(self.run_particle_tracking)
        self.gb_tracking.layout().addWidget(self.gb_tracking.btn_run, 3, 0, 1, 2)
        self.rpanel.layout().addWidget(self.gb_tracking)

        ########
        # Trace sorting
        self.gb_sorting = QtWidgets.QGroupBox('5. Particle tracking')
        self.gb_sorting.setLayout(QtWidgets.QGridLayout())
        # Toggle sorting
        self.gb_sorting.layout().addWidget(QtWidgets.QLabel('Toggle sorting'), 0, 0)
        self.gb_sorting.toggle_sorting = QtWidgets.QCheckBox()
        self.gb_sorting.toggle_sorting.setChecked(False)
        self.gb_sorting.toggle_sorting.clicked.connect(self.start_particle_id_sorting)
        self.gb_sorting.layout().addWidget(self.gb_sorting.toggle_sorting, 0, 1)
        self.rpanel.layout().addWidget(self.gb_sorting)
        # Save
        self.gb_sorting.btn_save = QtWidgets.QPushButton('Save')
        self.gb_sorting.btn_save.clicked.connect(self.save_particle_id_sorting)
        self.gb_sorting.btn_save.setEnabled(False)
        self.gb_sorting.layout().addWidget(self.gb_sorting.btn_save, 3, 0, 1, 2)
        # Attrs
        self.particle_id_map: dict = None
        self.particle_labels = dict()

        self.btn_test = QtWidgets.QPushButton('TEST')
        self.btn_test.clicked.connect(lambda: self.start_thread(TrackpyLinker))
        self.rpanel.layout().addWidget(self.btn_test)

        # Spacer
        vSpacer = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.rpanel.layout().addItem(vSpacer)

        # Add statusbar
        gv.statusbar = Statusbar()
        self.setStatusBar(gv.statusbar)

        # Start update timer
        self.ui_update_timer = QtCore.QTimer()
        self.ui_update_timer.setInterval(100)
        self.ui_update_timer.timeout.connect(self.update_ui)
        self.ui_update_timer.start()

        self.show()

################
# THREADING
    def start_thread(self, thread_class, *args, **kwargs):
        # Redirect stdout
        self.queue = Queue()
        self._stdout = sys.stdout
        sys.stdout = CustomStdout(self.queue)
        self.extern_dialog = ExternalWidget(self.queue)

        print('Start thread')
        self.thread = thread_class(*args, **kwargs)
        self.thread.finished.connect(self.terminate_thread)
        self.thread.start()

    def terminate_thread(self):
        # Re-route stdout
        sys.stdout = self._stdout

        self.extern_dialog.close()
        self.thread.terminate()

################
#
    def detection_diameter_validator(self, value):
        value = value + 1 if value % 2 == 0 else value
        self.gb_detection.spn_diameter.setValue(value)

    def update_ui(self):
        self.gb_display_switch.setEnabled(gv.dset is not None)

    @staticmethod
    def get_particle_color(particle_id, order):
        colors = gv.CMAP_COLORS

        alpha = 250 // order if order > 0 else 255

        if particle_id >= 0:
            c = (*[int(255 * c) for c in colors[int(particle_id) % len(colors)]], alpha)
        else:
            c = (255, 0, 0, alpha)

        return c

    def get_particle_id(self, p):
        # Get particle id if available
        particle_id = p.particle if 'particle' in p else -1

        # Map if applicable
        if self.particle_id_map is not None \
                and particle_id in self.particle_id_map:
            particle_id = self.particle_id_map[particle_id]

        return int(particle_id)

    def update_image(self):
        """Update HDF5Viewer"""
        im = self.viewer.get_image()

        if im is None:
            return

        self.viewer.set_image(im)

        # Remove all previous markers
        self.particle_markers.clear()

        # Plot particles
        if gv.tpf is None:
            return


        add_x = gv.h5f.attrs[gv.KEY_ATTR_ROI_POS][1] \
            if gv.dset.name == f'/{gv.KEY_ORIGINAL}' \
            else 0
        add_y = gv.h5f.attrs[gv.KEY_ATTR_ROI_POS][0] \
            if gv.dset.name == f'/{gv.KEY_ORIGINAL}' \
            else 0

        frame_idx = self.viewer.slider.value()

        # Display past particles
        particle_trace_max = 10
        spots = list()
        for i in range(1, particle_trace_max):
            idx = frame_idx-i
            try:
                rows = gv.tpf.get(idx).iterrows()
            except Exception as exc:
                rows = None

            if rows is None:
                continue

            for _, p in rows:
                particle_id = self.get_particle_id(p)

                spots.append({'pos': [p.y+add_y, p.x+add_x],
                               'data': (particle_id, i),
                               'size': particle_trace_max-i,
                               'pen': self.get_particle_color(particle_id, i),
                               'brush': None})

        # Plot current particle
        try:
            rows = gv.tpf.get(frame_idx).iterrows()
        except Exception as exc:
            rows = None

        if rows is None:
            return

        # Clear particle_id labels
        for id, label in self.particle_labels.items():
            label.setText('')

        # Display current points
        for _, p in rows:
            particle_id = self.get_particle_id(p)

            pos = [p.y+add_y, p.x+add_x]
            color = self.get_particle_color(particle_id, 0)

            spots.append({'pos': pos,
                           'data': (particle_id, 0),
                           'size': 10,
                           'pen': color,
                           'brush': None})

            # Add particle_id label
            if particle_id not in self.particle_labels:
                self.particle_labels[particle_id] = pg.TextItem(str(particle_id))
                self.viewer.view.addItem(self.particle_labels[particle_id])

            # Set particle_id and position
            self.particle_labels[particle_id].setText(str(particle_id))
            self.particle_labels[particle_id].setPos(*pos)

        # Add spots to scatter plot item
        self.particle_markers.addPoints(spots)

    def start_particle_id_sorting(self):
        self.particle_id_map = dict()
        self.gb_sorting.btn_save.setEnabled(True)

    def save_particle_id_sorting(self):
        print(self.particle_id_map)

        self.particle_id_map = None
        # TODO: iterate over frames, map ids and save to file
        self.gb_sorting.toggle_sorting.setChecked(False)

    def clicked_on_particle(self, obj, points):
        # Only if sorting is toggled
        if not(self.gb_sorting.toggle_sorting.isChecked()):
            return

        # Click on 1st particle: copy id
        if not(hasattr(self, 'current_particle_id')) \
                or self.current_particle_id is None:
            self.current_particle_id = points[0].data()[0]
            print(f'Set particle {self.current_particle_id}')
            return

        # Click on 2nd particle: overwrite with last id
        id = points[0].data()[0]
        confirm_dialog = QtWidgets.QMessageBox.question(
            gv.w, 'Overwrite?',
            f'Swap particles {id} and {self.current_particle_id}?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)

        if confirm_dialog == QtWidgets.QMessageBox.No:
            self.current_particle_id = None
            return

        print(f'Swap particles {id} and {self.current_particle_id}')
        self.particle_id_map[id] = self.current_particle_id
        self.current_particle_id = None

    def update_roi_display(self):
        if not(hasattr(self, 'rect_roi')):
            self.rect_roi = pg.RectROI([0,0], [1,1])
            self.rect_roi.sigRegionChangeFinished.connect(self.update_roi_params)
            self.viewer.view.addItem(self.rect_roi)

        if gv.dset is None:
            return

        # Hide for processed version within ROI
        if gv.dset.name == f'/{gv.KEY_PROCESSED}':
            self.viewer.view.removeItem(self.rect_roi)
            del self.rect_roi
            return

        if gv.KEY_ATTR_ROI_POS in gv.h5f.attrs and gv.KEY_ATTR_ROI_SIZE in gv.h5f.attrs:
            pos_ = gv.h5f.attrs[gv.KEY_ATTR_ROI_POS]
            size_ = gv.h5f.attrs[gv.KEY_ATTR_ROI_SIZE]
        else:
            pos_ = [0, 0]
            size_ = [gv.dset.shape[1], gv.dset.shape[2]]

        self.rect_roi.setPos(pg.Point(pos_))
        self.rect_roi.setSize(pg.Point(size_))

    @staticmethod
    def update_roi_params(roi: pg.RectROI):
        gv.h5f.attrs[gv.KEY_ATTR_ROI_POS] = [roi.pos().x(), roi.pos().y()]
        gv.h5f.attrs[gv.KEY_ATTR_ROI_SIZE] = [roi.size().x(), roi.size().y()]

    def run_roi_filter(self):

        if not(hasattr(self, 'rect_roi')):
            print(f'Please select {gv.KEY_ORIGINAL} dataset')
            return

        data = self.viewer.get_image()
        img = self.viewer.image_item
        im, coords = self.rect_roi.getArrayRegion(data, img, returnMappedCoords=True)

        coords = coords.astype(int)

        median_subtraction.run(self.gb_filter.seg_len.value(), self.gb_filter.med_range.value(), coords)

    def run_particle_detection(self):
        print('Run particle detection')

        invert = self.gb_detection.check_invert.isChecked()
        diameter = self.gb_detection.spn_diameter.value()
        minmass = self.gb_detection.spn_minmass.value()

        print(f'Invert {invert}')
        print(f'Diameter {diameter}')
        print(f'Minmass {minmass}')

        filepath = f'{gv.filepath}.{gv.EXT_TRACKPY}'

        # If detection file exists, ask if should be removed
        if os.path.exists(filepath):
            confirm_dialog = QtWidgets.QMessageBox.question(
                gv.w, 'Overwrite?',
                'Previous particle detection exists. Overwrite?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.No)

            if confirm_dialog \
                    in [QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Cancel]:
                return

            print('Remove previous detection file')
            gv.tpf.close()
            gv.tpf = None
            os.remove(filepath)

        # Run detection
        with tp.PandasHDFStoreBig(filepath) as s:
            tp.batch(gv.h5f[gv.KEY_PROCESSED], diameter, invert=invert, minmass=minmass, output=s, processes='auto')

        # Open detection file
        gv.tpf = tp.PandasHDFStoreBig(filepath)

    def run_particle_tracking(self):
        srange = self.gb_tracking.srange.value()
        smemory = self.gb_tracking.smemory.value()

        self.start_thread(TrackpyLinker, srange, smemory)

        # srange = self.gb_tracking.srange.value()
        # smemory = self.gb_tracking.smemory.value()
        #
        # with tp.PandasHDFStoreBig(f'{gv.filepath}.{gv.EXT_TRACKPY}') as s:
        #     pred = tp.predict.NearestVelocityPredict()
        #     for linked in pred.link_df_iter(s, srange, memory=smemory):
        #         s.put(linked)

    def set_title(self, sub=None):
        sub = ' - {}'.format(sub) if not(sub is None) else ''
        self.setWindowTitle('3D Annotator' + sub)

    def set_dataset(self, dset_name):
        if gv.h5f is None:
            gv.dset = None
        else:
            if not(dset_name in gv.h5f):
                print(f'WARNING: dataset {dset_name} not in file')
                return
            else:
                gv.dset = gv.h5f[dset_name]

        self.viewer.set_dataset(gv.dset)

        if dset_name == gv.KEY_ORIGINAL:
            gv.w.gb_display_switch.btn_raw.setStyleSheet('font-weight:bold;')
            gv.w.gb_display_switch.btn_processed.setStyleSheet('font-weight:normal;')
        elif dset_name == gv.KEY_PROCESSED:
            gv.w.gb_display_switch.btn_processed.setStyleSheet('font-weight:bold;')
            gv.w.gb_display_switch.btn_raw.setStyleSheet('font-weight:normal;')
        else:
            gv.w.gb_display_switch.btn_raw.setStyleSheet('font-weight:normal;')
            gv.w.gb_display_switch.btn_processed.setStyleSheet('font-weight:normal;')

        self.update_roi_display()

        self.viewer.slider.valueChanged.emit(self.viewer.slider.value())


class CustomStdout:

    def __init__(self, queue):
        self.queue = queue

    def write(self, string):
        self.queue.put(string)

class TrackpyLinker(QtCore.QThread):

    def __init__(self, srange, smemory):
        QtCore.QThread.__init__(self)
        self.srange = srange
        self.smemory = smemory

    def run(self):
        print('Start TrackpyLinker')

        with tp.PandasHDFStoreBig(f'{gv.filepath}.{gv.EXT_TRACKPY}') as s:
            pred = tp.predict.NearestVelocityPredict()
            for linked in pred.link_df_iter(s, self.srange, memory=self.smemory):
                s.put(linked)


class ExternalWidget(QtWidgets.QWidget):

    def __init__(self, queue):
        QtWidgets.QWidget.__init__(self)
        self.queue = queue
        self.setLayout(QtWidgets.QHBoxLayout())
        self.txt = QtWidgets.QPlainTextEdit()
        self.layout().addWidget(self.txt)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.write)
        self.timer.start(100)

        self.resize(500,500)
        self.show()

    def write(self):
        while not(self.queue.empty()):
            self.txt.appendPlainText(self.queue.get().replace('\n', ''))

################################
# Statusbar widget

class Statusbar(QtWidgets.QStatusBar):
    def __init__(self):
        QtWidgets.QStatusBar.__init__(self)

        self.setReady()
        self.progressbar = QtWidgets.QProgressBar()
        self.progressbar.setMaximumWidth(300)

        self.addPermanentWidget(self.progressbar)
        self.progressbar.hide()

    def startBlocking(self, msg):
        self.setEnabled(False)
        self.showMessage(msg)
        gv.app.processEvents()

    def setReady(self):
        self.setEnabled(True)
        self.showMessage('Ready')
        gv.app.processEvents()

    def startProgress(self, descr, max_value):
        self.progressbar.show()
        self.showMessage(descr)
        self.progressbar.setMaximum(max_value)
        gv.w.setEnabled(False)
        gv.app.processEvents()

    def setProgress(self, value):
        self.progressbar.setValue(value)
        gv.app.processEvents()

    def endProgress(self):
        self.progressbar.hide()
        self.showMessage('Ready')
        gv.w.setEnabled(True)
        gv.app.processEvents()



################################
### Particle detection widget

class ParticleDetectionWidget(QtWidgets.QGroupBox):

    def __init__(self, *args):
        QtWidgets.QGroupBox.__init__(self, *args)
        ### Checkstate
        self.setCheckable(True)
        self.setChecked(False)

        ### Layout

        ### Threshold widget

        self.thresh_rule = QtWidgets.QComboBox()
        self.thresh_rule.addItems(['>', '<'])
        self.thresh_rule.setEnabled(False)
        self.layout().addWidget(QLabel('Threshold rule'), 0, 0)
        self.layout().addWidget(self.thresh_rule, 0, 1)

        self.std_mult = QtWidgets.QDoubleSpinBox()
        self.std_mult.setMinimum(0.01)
        self.std_mult.setSingleStep(0.01)
        self.std_mult.setValue(2.5)
        self.layout().addWidget(QLabel('SD multiplier'), 2, 0)
        self.layout().addWidget(self.std_mult, 2, 1)



################################
### HDF5ImageView
"""
HDF5ImageView is an image view for displaying memory-mapped image sequences from HDF5 formatted files in pyqtgraph.

Adapted in part from pyqtgraph's ImageView:
> ImageView.py -  Widget for basic image dispay and analysis
> Copyright 2010  Luke Campagnola
> Distributed under MIT/X11 license. See license.txt for more information.

2020 Tim Hladnik
"""


class HDF5ImageView(QtWidgets.QWidget):

    def __init__(self, *args, update_fun=None, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.vwdgt = QtWidgets.QWidget(self)
        self.vwdgt.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.vwdgt)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.layout().addWidget(self.slider)
        if update_fun is None:
            self.slider.valueChanged.connect(self.update_image)
        else:
            self.slider.valueChanged.connect(update_fun)

        ### Viewbox
        self.view = pg.ViewBox()
        #self.view.setMouseEnabled(False, False)
        self.view.setAspectLocked(True)

        ### Graphics view
        self.graphicsView = pg.GraphicsView(self.vwdgt)
        self.vwdgt.layout().addWidget(self.graphicsView)
        self.graphicsView.setCentralItem(self.view)

        ### Scene
        self.scene = self.graphicsView.scene()

        ### Image item
        self.image_item = pg.ImageItem()
        self.view.addItem(self.image_item)

        #self.imageItem.setImage(np.random.randint(10, size=(500, 600)))
        self.playTimer = QtCore.QTimer()
        self.playTimer.timeout.connect(self.timeout)

        self.noRepeatKeys = [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown, QtCore.Qt.Key_A, QtCore.Qt.Key_D]
        self.keysPressed = dict()
        self._filters = dict()
        self.playRate = 0

    def set_dataset(self, dset):
        self.dset = dset

        # TODO: reset range when dataset is changed
        #self.graphicsView.enableMouse(False)
        #self.graphicsView.updateMatrix()
        #import IPython
        #IPython.embed()
        #self.graphicsView.setRange(disableAutoPixel=False)

        if self.dset is None:
            self.slider.setEnabled(False)
            return

        self.slider.setEnabled(True)
        z_len = self.dset.shape[0]-1
        self.slider.setMinimum(0)
        self.slider.setMaximum(z_len)
        self.slider.setTickInterval(1//(100/z_len))
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.update_image()


    def get_image(self):
        if self.dset is None:
            return None

        return self.dset[self.slider.value(),:,:]

    def set_image(self, im):
        self.image_item.setImage(im)

    def update_image(self):

        im = self.get_image()

        if im is None:
            im = np.array([[[0]]])

        self.image_item.setImage(im)

    def play(self, rate=None):
        """Begin automatically stepping frames forward at the given rate (in fps).
        This can also be accessed by pressing the spacebar."""
        # print "play:", rate
        if rate is None:
            rate = 10
        self.playRate = rate

        if rate == 0:
            self.playTimer.stop()
            return

        self.lastPlayTime = time.time()
        if not self.playTimer.isActive():
            self.playTimer.start(16)

    def timeout(self):
        now = time.time()
        dt = now - self.lastPlayTime
        if dt < 0:
            return

        n = int(self.playRate * dt)
        if n != 0:
            self.lastPlayTime += (float(n)/self.playRate)
            if self.slider.value()+n > self.dset.shape[0]:
                self.play(0)
            self.jumpFrames(n)

    def jumpFrames(self, n):
        """Move video frame ahead n frames (may be negative)"""
        self.slider.setValue(self.slider.value()+n)

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Space:
            if self.playRate == 0:
                self.play()
            else:
                self.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_1:
            self.play(5)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_2:
            self.play(10)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_3:
            self.play(20)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_Home:
            self.slider.setValue(0)
            self.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_End:
            self.slider.setValue(self.dset.shape[0] - 1)
            self.play(0)
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.keysPressed[ev.key()] = 1
            self.evalKeyState()
        else:
            QtWidgets.QWidget.keyPressEvent(self, ev)

    def keyReleaseEvent(self, ev):
        if ev.key() in [QtCore.Qt.Key_Space, QtCore.Qt.Key_Home, QtCore.Qt.Key_End]:
            ev.accept()
        elif ev.key() in self.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            try:
                del self.keysPressed[ev.key()]
            except:
                self.keysPressed = {}
            self.evalKeyState()
        else:
            QtWidgets.QWidget.keyReleaseEvent(self, ev)

    def evalKeyState(self):
        if len(self.keysPressed) == 1:
            key = list(self.keysPressed.keys())[0]
            if key == QtCore.Qt.Key_Right or key == QtCore.Qt.Key_D:
                self.play(20)
                self.jumpFrames(1)
                self.lastPlayTime = time.time() + 0.2  ## 2ms wait before start
                ## This happens *after* jumpFrames, since it might take longer than 2ms
            elif key == QtCore.Qt.Key_Left or key == QtCore.Qt.Key_A:
                self.play(-20)
                self.jumpFrames(-1)
                self.lastPlayTime = time.time() + 0.2
            elif key == QtCore.Qt.Key_Up:
                self.play(-100)
            elif key == QtCore.Qt.Key_Down:
                self.play(100)
            elif key == QtCore.Qt.Key_PageUp:
                self.play(-1000)
            elif key == QtCore.Qt.Key_PageDown:
                self.play(1000)
        else:
            self.play(0)

################################################################
### Main

if __name__ == '__main__':
    ### Create application
    gv.app = QtWidgets.QApplication([])

    #gvars.open_dir = './testdata'
    gv.open_dir = 'T:'

    ################################
    ### Setup colormap for markers

    colormap = cm.get_cmap("tab20")
    colormap._init()
    cmap_lut = np.array((colormap._lut * 255))
    gv.cmap_lut = np.append(cmap_lut[::2, :], cmap_lut[1::2, :], axis=0)

    ################
    ### Create window

    gv.w = MainWindow()

    gv.app.exec_()

    file_handling.close_file()

