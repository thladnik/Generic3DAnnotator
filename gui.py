import numpy as np
from matplotlib import cm
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QLabel
import pyqtgraph as pg
from scipy.spatial import distance
from sklearn.neighbors import NearestCentroid
import time

import gv
import file_handling
import object_handling
import processing
import axis_calibration
import median_subtraction
import particle_detection


################################
### Main window

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        gv.w = self

        self.resize(1300, 1000)
        self.setTitle()
        self.cw = QtWidgets.QWidget()
        self.setCentralWidget(self.cw)
        self.cw.setLayout(QtWidgets.QGridLayout())
        
        ################
        ### Create menu
        
        self.mb = QtWidgets.QMenuBar()
        self.setMenuBar(self.mb)
        self.mb_file = self.mb.addMenu('File')
        self.mb_file_open = self.mb_file.addAction('Open file...')
        self.mb_file_open.setShortcut('Ctrl+O')
        self.mb_file_open.triggered.connect(file_handling.open_file)
        self.mb_file_import = self.mb_file.addAction('Import image sequence...')
        self.mb_file_close = self.mb_file.addAction('Close file')
        self.mb_file_close.triggered.connect(file_handling.close_file)
        self.mb_file_import.setShortcut('Ctrl+I')
        self.mb_file_import.triggered.connect(file_handling.import_file)
        self.mb_edit = self.mb.addMenu('Edit')
        self.mb_edit.act_rot_90cw = self.mb_edit.addAction('Rotate 90° CW')
        self.mb_edit.act_rot_90cw.triggered.connect(lambda: self.updateRotationFilter(-1))
        self.mb_edit.act_rot_90ccw = self.mb_edit.addAction('Rotate 90° CCW')
        self.mb_edit.act_rot_90ccw.triggered.connect(lambda: self.updateRotationFilter(1))
        self.mb_edit.act_rot_180 = self.mb_edit.addAction('Rotate 180°')
        self.mb_edit.act_rot_180.triggered.connect(lambda: self.updateRotationFilter(2))
        
        ################
        ### Create ImageView
        self.viewer = HDF5ImageView(self)
        self.viewer.scene.sigMouseClicked.connect(object_handling.add_node)
        self.viewer.scene.sigMouseClicked.connect(axis_calibration.set_axis_point)
        self.cw.layout().addWidget(self.viewer, 0, 0)
        self.viewer.slider.valueChanged.connect(object_handling.update_pos_marker)
        self.viewer.objects = dict()
        
        ################
        ### Create right panel
        
        self.rpanel = QtWidgets.QWidget()
        self.rpanel.setEnabled(False)
        self.rpanel.setFixedWidth(500)
        self.rpanel.setLayout(QtWidgets.QVBoxLayout())
        self.cw.layout().addWidget(self.rpanel, 0, 1)
        
        ########
        ### Display
        self.gb_display = QtWidgets.QGroupBox('Display')
        self.gb_display.setLayout(QtWidgets.QVBoxLayout())
        self.gb_display.btn_raw = QtWidgets.QPushButton('Raw')
        self.gb_display.btn_raw.clicked.connect(lambda: self.setDataset(gv.KEY_ORIGINAL))
        self.gb_display.layout().addWidget(self.gb_display.btn_raw)
        self.gb_display.btn_processed = QtWidgets.QPushButton('Processed')
        self.gb_display.btn_processed.clicked.connect(lambda: self.setDataset(gv.KEY_PROCESSED))
        self.gb_display.layout().addWidget(self.gb_display.btn_processed)
        self.rpanel.layout().addWidget(self.gb_display)
        
        ########
        ### Calibration
        ## Axes
        self.gb_calib = QtWidgets.QGroupBox('Calibration')
        self.gb_calib.setLayout(QtWidgets.QVBoxLayout())
        # Add label
        self.gb_calib.lbl_axes = QLabel('Axes for scaling')
        self.gb_calib.lbl_axes.setStyleSheet('font-weight:bold;')
        self.gb_calib.layout().addWidget(self.gb_calib.lbl_axes)
        # Add Button
        self.gb_calib.btn_add_axes = QtWidgets.QPushButton('')
        self.gb_calib.btn_add_axes.clicked.connect(axis_calibration.start_axes_calibration)
        self.viewer.slider.valueChanged.connect(
            lambda: self.gb_calib.btn_add_axes.setText('Set axes for frame {}'.format(self.viewer.slider.value())))
        self.gb_calib.layout().addWidget(self.gb_calib.btn_add_axes)
        # Status indicator
        self.gb_calib.le_axes_status = QtWidgets.QLineEdit('')
        self.gb_calib.le_axes_status.setEnabled(False)
        self.gb_calib.le_axes_status.setStyleSheet('font-weight:bold; color:orange;')
        self.gb_calib.layout().addWidget(self.gb_calib.le_axes_status)
        # Table
        self.gb_calib.axes_table = QtWidgets.QTableWidget()
        self.gb_calib.axes_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.gb_calib.layout().addWidget(self.gb_calib.axes_table)
        # Clear button
        self.gb_calib.btn_clear_axes = QtWidgets.QPushButton('Clear calibration')
        self.gb_calib.btn_clear_axes.clicked.connect(axis_calibration.clear_axes_calibration)
        self.gb_calib.layout().addWidget(self.gb_calib.btn_clear_axes)
        
        # Markers
        self.viewer.slider.valueChanged.connect(axis_calibration.update_axes_marker)
        
        ## Reference
        # Label
        self.gb_calib.lbl_ref = QLabel('References')
        self.gb_calib.lbl_ref.setStyleSheet('font-weight:bold;')
        self.gb_calib.layout().addWidget(self.gb_calib.lbl_ref)
        # Button
        self.gb_calib.btn_add_ref = QtWidgets.QPushButton('')
        # gb_calib.btn_add_ref.clicked.connect(start_ref_calibration)
        self.viewer.slider.valueChanged.connect(
            lambda: self.gb_calib.btn_add_ref.setText('Set reference for frame {}'.format(self.viewer.slider.value())))
        self.gb_calib.layout().addWidget(self.gb_calib.btn_add_ref)
        
        # Add to centralwidget
        self.rpanel.layout().addWidget(self.gb_calib)


        ########
        ### Median subtration + range normalization
        self.gb_med_norm = QtWidgets.QGroupBox('Median subtraction + Range Normalization')
        self.gb_med_norm.setLayout(QtWidgets.QGridLayout())
        self.gb_med_norm.setCheckable(True)
        self.gb_med_norm.setChecked(False)
        self.gb_med_norm.toggled.connect(self.updateMedianSubtractionFilter)
        ## Segment length
        self.gb_med_norm.seg_len = QtWidgets.QSpinBox()
        self.gb_med_norm.seg_len.setMinimum(1)
        self.gb_med_norm.seg_len.setValue(40)
        self.gb_med_norm.layout().addWidget(QLabel('Seg. len. [lower = less RAM]'), 0, 0)
        self.gb_med_norm.layout().addWidget(self.gb_med_norm.seg_len, 0, 1)
        ## Median range
        self.gb_med_norm.med_range = QtWidgets.QSpinBox()
        self.gb_med_norm.med_range.setMinimum(2)
        self.gb_med_norm.med_range.setValue(20)
        self.gb_med_norm.med_range.valueChanged.connect(self.updateMedianSubtractionFilter)
        self.gb_med_norm.layout().addWidget(QLabel('Range [lower = faster]'), 1, 0)
        self.gb_med_norm.layout().addWidget(self.gb_med_norm.med_range, 1, 1)
        ## Run
        self.gb_med_norm.btn_run = QtWidgets.QPushButton('Run subtraction + normalization')
        median_range = self.gb_med_norm.med_range.value
        segment_length = self.gb_med_norm.seg_len.value
        self.gb_med_norm.btn_run.clicked.connect(lambda: median_subtraction.run(segment_length(), median_range()))
        self.gb_med_norm.layout().addWidget(self.gb_med_norm.btn_run, 2, 0, 1, 2)
        self.rpanel.layout().addWidget(self.gb_med_norm)


        ########
        ### Particle detection
        self.gb_part_detect = ParticleDetectionWidget('Particle detection', self)
        self.rpanel.layout().addWidget(self.gb_part_detect)

        ########
        ### Objects panel
        self.gb_objects = QtWidgets.QGroupBox('Objects')
        self.gb_objects.setLayout(QtWidgets.QVBoxLayout())
        # New object
        self.gb_objects.btn_new_object = QtWidgets.QPushButton('New object')
        self.gb_objects.btn_new_object.clicked.connect(object_handling.create_object)
        self.gb_objects.layout().addWidget(self.gb_objects.btn_new_object)
        self.gb_objects.wdgt_buttons = QtWidgets.QWidget()
        self.gb_objects.wdgt_buttons.setLayout(QtWidgets.QVBoxLayout())
        self.gb_objects.layout().addWidget(self.gb_objects.wdgt_buttons)
        self.rpanel.layout().addWidget(self.gb_objects)
        
        vSpacer = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.rpanel.layout().addItem(vSpacer)
        
        ### Add statusbar
        gv.statusbar = Statusbar()
        self.setStatusBar(gv.statusbar)

        self.viewer.slider.valueChanged.connect(self.updateParticleMarkers)

        ### Set up window and open/execute
        self.show()

    def setTitle(self, sub=None):
        sub = ' - {}'.format(sub) if not(sub is None) else ''
        self.setWindowTitle('3D Annotator' + sub)


    def updateMedianSubtractionFilter(self):
        if self.gb_med_norm.isChecked():
            self.viewer.addImageFilter('median_sub', processing.median_norm_filter, 5, [self.gb_med_norm.med_range.value()])

        else:
            self.viewer.removeImageFilter('median_sub')

    def updateRotationFilter(self, dir):
        if not(gv.KEY_ROT in gv.f.attrs):
            gv.f.attrs[gv.KEY_ROT] = 0
        gv.f.attrs[gv.KEY_ROT] += dir

        self.viewer.removeImageFilter('rotation')
        self.viewer.addImageFilter('rotation', np.rot90, 1, [gv.f.attrs[gv.KEY_ROT]])
        self.viewer.slider.valueChanged.emit(self.viewer.slider.value())


    def setDataset(self, dsetName):
        if gv.f is None:
            gv.dset = None
            #self.viewer.setDataset(None)
        else:
            if not(dsetName in gv.f):
                print('WARNING: dataset {} not in file'.format(dsetName))
                return
            else:
                gv.dset = gv.f[dsetName]
                #self.viewer.setDataset(gv.f[dsetName])
        self.viewer.setDataset(gv.dset)

        if dsetName == gv.KEY_ORIGINAL:
            gv.w.gb_display.btn_raw.setStyleSheet('font-weight:bold;')
            gv.w.gb_display.btn_processed.setStyleSheet('font-weight:normal;')
        elif dsetName == gv.KEY_PROCESSED:
            gv.w.gb_display.btn_processed.setStyleSheet('font-weight:bold;')
            gv.w.gb_display.btn_raw.setStyleSheet('font-weight:normal;')
        else:
            gv.w.gb_display.btn_raw.setStyleSheet('font-weight:normal;')
            gv.w.gb_display.btn_processed.setStyleSheet('font-weight:normal;')

        self.viewer.slider.valueChanged.emit(self.viewer.slider.value())

    def updateParticleMarkers(self):
        if not(hasattr(self.viewer, 'particles')):
            self.viewer.particles = dict()

        ### Check if particles are correctly saved to file
        if gv.f is None \
                or not(gv.KEY_PARTICLES in gv.f) \
                or not(gv.KEY_PART_AREA in gv.f) \
                or not(gv.KEY_PART_CENTR in gv.f):
            return

        self.updateParticle2ObjectMatches()

        ### Display particle markers
        frame_idx = self.viewer.slider.value()
        i = None
        for i, centroid in enumerate(gv.f[gv.KEY_PART_CENTR][frame_idx,:,:]):
            ### If centroid not displayed in this frame: skip
            if np.any(np.isnan(centroid)):

                ### If it was displayed in previous frame: remove and delete
                if i in self.viewer.particles:
                    self.viewer.view.removeItem(self.viewer.particles[i])
                    del self.viewer.particles[i]

                continue

            ### Create new marker
            if not(i in self.viewer.particles):


                ## Create marker
                marker = pg.PlotDataItem(x=[centroid[0]], y=[centroid[1]], symbolBrush=None, symbolPen=None, symbol='o', symbolSize=18,
                                   name='centroid_{}'.format(i))
                self.viewer.particles[i] = marker
                self.viewer.view.addItem(marker)

            ### Set color
            obj_name = '{}{}'.format(gv.KEY_OBJSTR, i)
            if gv.KEY_OBJLIST in gv.f.attrs and obj_name in gv.f.attrs[gv.KEY_OBJLIST] \
                    and np.all(np.isfinite(gv.f[obj_name][gv.KEY_NODEINTERP][frame_idx,:])):
                pen = pg.mkPen((*gv.cmap_lut[i, :3],255,), width=2)
            else:
                pen = pg.mkPen((255,255,255,255,), width=1, style=QtCore.Qt.DotLine)
            self.viewer.particles[i].setSymbolPen(pen)

            ### Set coordinates of particle centroid
            self.viewer.particles[i].setData(x=[centroid[0]], y=[centroid[1]])


    def updateParticle2ObjectMatches(self):
        #print('update1')

        ### Check if particles and objects are correctly saved to file
        if gv.f is None \
                or not(gv.KEY_PARTICLES in gv.f) \
                or not(gv.KEY_PART_AREA in gv.f) \
                or not(gv.KEY_PART_CENTR in gv.f) \
                or not(gv.KEY_OBJLIST in gv.f.attrs) \
                or len(gv.f.attrs[gv.KEY_OBJLIST]) == 0:
            return
        #print('update2')

        frame_idx = self.viewer.slider.value()
        ### Calculate particle distances to objects
        obj_pos = np.array([gv.f[obj_name][gv.KEY_NODEINTERP][frame_idx,:] for obj_name in gv.f.attrs[gv.KEY_OBJLIST]])
        particles = gv.f[gv.KEY_PART_CENTR][frame_idx,:,:]

        if particles.shape[0] < obj_pos.shape[0]:
            print('WARNING: number of particles in frame {} smaller than number of objects'.format(frame_idx))
            return

        ### Filter all non-NaN objects for this frame
        obj_idcs = np.arange(obj_pos.shape[0], dtype=int)[np.isfinite(obj_pos[:,0])]
        particles = particles[np.isfinite(particles[:,0]),:]


        use_exclusive_matching = True
        if use_exclusive_matching:
            ### Classify particles
            if obj_idcs.shape[0] > 1:
                clf = NearestCentroid()
                clf.fit(obj_pos[obj_idcs,:], obj_idcs)
                result = clf.predict(particles)
            elif obj_idcs.shape[0] == 1:
                result = obj_idcs[0] * np.ones(particles.shape[0])
            else:
                return

            winners = list()
            rejects = list()
            for i, pos in enumerate(obj_pos):
                pot_particles = particles[result == i]
                if pot_particles.shape[0] == 0:
                    winners.append([np.nan, np.nan])
                    continue

                dists = [(distance.euclidean(pos, x), j) for j, x in enumerate(pot_particles)]
                if len(dists) == 0:
                    continue

                dists = sorted(dists)
                winners.append(pot_particles[dists[0][1]])
                if len(dists) > 1:
                    rejects.extend([pot_particles[d[1]] for d in dists[1:]])

            all_particles = np.array([*winners, *rejects])

        else:
            pass

        #print(len(combined), len(particles))

        #print(combined)

        #import IPython
        #IPython.embed()
        gv.f[gv.KEY_PART_CENTR][frame_idx,:,:] = np.nan
        gv.f[gv.KEY_PART_CENTR][frame_idx,:all_particles.shape[0],:] = all_particles





################################
### Statusbar widget

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
### Threshold widget

class ThresholdWidget(QtWidgets.QGroupBox):
    
    def __init__(self, *args):
        QtWidgets.QGroupBox.__init__(self, *args)

        self.setCheckable(True)
        self.setChecked(False)
        self.setLayout(QtWidgets.QGridLayout())
        # Threshold
        self.thresh = QtWidgets.QSpinBox()
        self.thresh.setMinimumWidth(0)
        self.thresh.setMaximum(2 ** 8 - 1)
        self.thresh.setValue(120)
        self.layout().addWidget(QLabel('Threshold'), 0, 0)
        self.layout().addWidget(self.thresh, 0, 1)
        # Maxval
        self.maxval = QtWidgets.QSpinBox()
        self.maxval.setMinimumWidth(0)
        self.maxval.setMaximum(2 ** 8 - 1)
        self.maxval.setValue(2 ** 8 - 1)
        self.layout().addWidget(QLabel('Maxval'), 1, 0)
        self.layout().addWidget(self.maxval, 1, 1)
        # Type
        self.threshtype = QtWidgets.QComboBox()
        self.threshtype.addItems(
            ['THRESH_BINARY_INV', 'THRESH_BINARY', 'THRESH_TRUNC', 'THRESH_TOZERO_INV', 'THRESH_TOZERO'])
        self.layout().addWidget(QLabel('Type'), 2, 0)
        self.layout().addWidget(self.threshtype, 2, 1)
    
        ## Adaptive treshold
        self.gb_adaptive = QtWidgets.QGroupBox('Adaptive')
        self.gb_adaptive.setLayout(QtWidgets.QGridLayout())
        self.gb_adaptive.setCheckable(True)
        self.gb_adaptive.setChecked(False)
        self.gb_adaptive.method = QtWidgets.QComboBox()
        self.gb_adaptive.method.addItems(['ADAPTIVE_THRESH_MEAN_C', 'ADAPTIVE_THRESH_GAUSSIAN_C'])
        self.gb_adaptive.layout().addWidget(QLabel('Method'), 0, 0)
        self.gb_adaptive.layout().addWidget(self.gb_adaptive.method, 0, 1)
        self.gb_adaptive.block_size = QtWidgets.QSpinBox()
        self.gb_adaptive.block_size.setMinimum(3)
        self.gb_adaptive.block_size.setMaximum(500)
        self.gb_adaptive.block_size.setSingleStep(2)
        self.gb_adaptive.block_size.setValue(11)
        self.gb_adaptive.layout().addWidget(QLabel('Block size'), 1, 0)
        self.gb_adaptive.layout().addWidget(self.gb_adaptive.block_size, 1, 1)
        self.gb_adaptive.constant = QtWidgets.QSpinBox()
        self.gb_adaptive.constant.setMinimum(0)
        self.gb_adaptive.constant.setMaximum(2 ** 8 - 1)
        self.gb_adaptive.constant.setValue(5)
        self.gb_adaptive.layout().addWidget(QLabel('Constant'), 2, 0)
        self.gb_adaptive.layout().addWidget(self.gb_adaptive.constant, 2, 1)
        # Add adaptive
        self.layout().addWidget(self.gb_adaptive, 3, 0, 1, 2)

        # Connect events
        self.toggled.connect(self.updateThresholdFilter)
        self.thresh.valueChanged.connect(self.updateThresholdFilter)
        self.maxval.valueChanged.connect(self.updateThresholdFilter)
        self.threshtype.currentTextChanged.connect(self.updateThresholdFilter)
        self.gb_adaptive.toggled.connect(self.updateThresholdFilter)
        self.gb_adaptive.method.currentTextChanged.connect(self.updateThresholdFilter)
        self.gb_adaptive.block_size.valueChanged.connect(self.updateThresholdFilter)
        self.gb_adaptive.constant.valueChanged.connect(self.updateThresholdFilter)

    def getThresholdFilter(self):
        if self.gb_thresh.gb_adaptive.isChecked():
            fun = processing.adaptive_threshold_filter
            args = [self.maxval.value(),
                    self.gb_adaptive.method.currentText(),
                    self.threshtype.currentText(),
                    self.gb_adaptive.block_size.value(),
                    self.gb_adaptive.constant.value()]

        else:
            fun = processing.threshold_filter
            args = [self.thresh.value(),
                    self.maxval.value(),
                    self.threshtype.currentText()]

        return fun, args


    def updateThresholdFilter(self):
        if self.gb_thresh.isChecked():

            fun, args = self.getThresholdFilter()

            gv.w.viewer.addImageFilter('threshold', fun, 11, args)

        else:
            gv.w.viewer.removeImageFilter('threshold')

################################
### Particle detection widget

class ParticleDetectionWidget(QtWidgets.QGroupBox):

    def __init__(self, *args):
        QtWidgets.QGroupBox.__init__(self, *args)
        ### Checkstate
        self.setCheckable(True)
        self.toggled.connect(self.updateParticleDetectionFilter)
        self.setChecked(False)

        ### Layout
        self.setLayout(QtWidgets.QGridLayout())

        ### Threshold widget

        self.thresh_rule = QtWidgets.QComboBox()
        self.thresh_rule.addItems(['>', '<'])
        self.thresh_rule.currentTextChanged.connect(self.updateParticleDetectionFilter)
        self.layout().addWidget(QLabel('Threshold rule'), 0, 0)
        self.layout().addWidget(self.thresh_rule, 0, 1)

        self.std_mult = QtWidgets.QDoubleSpinBox()
        self.std_mult.setMinimum(0.01)
        self.std_mult.setSingleStep(0.01)
        self.std_mult.setValue(2.5)
        self.std_mult.valueChanged.connect(self.updateParticleDetectionFilter)
        self.layout().addWidget(QLabel('SD multiplier'), 2, 0)
        self.layout().addWidget(self.std_mult, 2, 1)

        self.btn_run = QtWidgets.QPushButton('Run particle detection')
        self.btn_run.clicked.connect(particle_detection.run)
        self.layout().addWidget(self.btn_run, 3, 0, 1, 2)


    def updateParticleDetectionFilter(self):
        if self.isChecked():
            gv.w.viewer.addImageFilter('particle_detection', processing.particle_filter, 20,
                                       [self.thresh_rule.currentText(), self.std_mult.value()])
        else:
            gv.w.viewer.removeImageFilter('particle_detection')


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

    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.vwdgt = QtWidgets.QWidget(self)
        self.vwdgt.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.vwdgt)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.layout().addWidget(self.slider)
        self.slider.valueChanged.connect(self.updateImage)

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
        self.imageItem = pg.ImageItem()
        self.view.addItem(self.imageItem)

        #self.imageItem.setImage(np.random.randint(10, size=(500, 600)))
        self.playTimer = QtCore.QTimer()
        self.playTimer.timeout.connect(self.timeout)

        self.noRepeatKeys = [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down, QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown, QtCore.Qt.Key_A, QtCore.Qt.Key_D]
        self.keysPressed = dict()
        self._filters = dict()
        self.playRate = 0


    def setDataset(self, dset):
        self.dset = dset

        if self.dset is None:
            self.slider.setEnabled(False)
            return

        self.slider.setEnabled(True)
        z_len = self.dset.shape[0]-1
        self.slider.setMinimum(0)
        self.slider.setMaximum(z_len)
        self.slider.setTickInterval(1//(100/z_len))
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.updateImage()

    def updateImage(self):
        if self.dset is None:
            self.imageItem.setImage(np.array([[[0]]]))
            return

        filters = [filt for filt in self._filters.items()]

        im = self.dset[self.slider.value(),:,:,:]
        if bool(filters):
            im = im.copy()

            for _, (filt_ord, filt_fun, filt_args) in sorted(filters):
                im = filt_fun(im, *filt_args)

        self.imageItem.setImage(im)

    def addImageFilter(self, filt_name, filt_fun, filt_order, filt_args):
        if filt_name in self._filters:
            self.removeImageFilter(filt_name)

        self._filters[filt_name] = (filt_order, filt_fun, filt_args,)

        self.updateImage()

    def removeImageFilter(self, filt_name):
        if not(filt_name in self._filters):
            return

        del self._filters[filt_name]
        self.updateImage()

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


    ### Calibration
    # Axes
    axes_order = ['xmin', 'xmax', 'ymin', 'ymax']
    set_axes = False
    axes_markers = dict()


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

