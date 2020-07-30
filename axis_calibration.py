from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QLabel
import pyqtgraph as pg
import numpy as np

import gv


################################################################
### Axes calibration

def start_axes_calibration():
    global set_axes, axes_order, gb_calib

    gv.w.viewer.slider.setEnabled(False)
    gb_calib.le_axes_status.setText('Set axis {}'.format(axes_order[0]))

    set_axes = True

def set_axis_point(ev):
    global set_axes, axes_order, gb_calib

    if not(ev.double()) or gv.set_axes is False:
        return

    ### Check if position is within limits
    pos = gv.w.viewer.view.mapSceneToView(ev.scenePos())
    x = pos.x()
    y = pos.y()

    if x > gv.f[gv.KEY_ORIGINAL].shape[1] or x < 0.0 or y > gv.f[gv.KEY_ORIGINAL].shape[2] or y < 0.0:
        print('Position [{}/{}] out of bounds.'.format(x,y))
        return


    ### Create datasets
    if 'axis_calibration_indices' not in gv.f:
        gv.f.create_dataset('axis_calibration_indices',
                            shape = (0, 1),
                            maxshape = (None, 1),
                            dtype = np.uint64)
    fidx_dset = gv.f['axis_calibration_indices']

    if 'axis_calibration_limits' not in gv.f:
        lim_dset = gv.f.create_dataset('axis_calibration_limits',
                                       shape = (0, 4, 2),
                                       maxshape = (None, 4, 2),
                                       dtype = np.float64,
                                       fillvalue = -1.0)
        ### Save calibration order
        lim_dset.attrs['axis_limit_order'] = axes_order
    lim_dset = gv.f['axis_calibration_limits']

    if 'axis_calibration_scale' not in gv.f:
        gv.f.create_dataset('axis_calibration_scale',
                            shape = (0, 4),
                            maxshape = (None, 4),
                            dtype = np.float64,
                            fillvalue = -1.0)
    scale_dset = gv.f['axis_calibration_scale']

    fidx = gv.w.viewer.slider.value()
    # Calibration for new frame
    if not(fidx in fidx_dset[:]):
        print('Start calibration for frame {}'.format(fidx))
        ### Resize datasets and set frame index
        fidx_dset.resize((fidx_dset.shape[0]+1, *fidx_dset.shape[1:]))
        fidx_dset[-1] = fidx
        lim_dset.resize((lim_dset.shape[0]+1, *lim_dset.shape[1:]))
        scale_dset.resize((scale_dset.shape[0]+1, *scale_dset.shape[1:]))

    ### Get current index
    idx = np.where(fidx_dset[:,0] == fidx)[0][0]

    # If all axis limits have been set already: reset them
    if np.sum(lim_dset[idx,:,0] >= 0) == lim_dset[:].shape[1]:
        print('Reset calbration for frame {}'.format(fidx))
        lim_dset[idx,:] = -1.0


    pidx = sum(lim_dset[idx,:,0] >= 0)
    lim_dset[idx,pidx,:] = [x,y]

    ### Set current axis limit
    print('New calibration point {}:{} [{}/{}] added for frame {}'.format(pidx, axes_order[pidx], x, y, fidx))
    update_axes_table()

    ### If last point was calibrated: ask for scale, update all and exit calibration
    if pidx == len(axes_order)-1:
        print('Calibration completed for frame {}'.format(fidx))
        set_axes = False

        gb_calib.le_axes_status.setText('Set scale')
        update_axes_marker()

        dialog = QtWidgets.QDialog(gv.w)
        dialog.setWindowTitle('Set scale')
        dialog.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)
        dialog.setLayout(QtWidgets.QGridLayout())
        dialog.lbl_set = QLabel('Set scale')
        dialog.lbl_set.setStyleSheet('font-weight:bold; text-alignment:center;')
        dialog.layout().addWidget(dialog.lbl_set, 0, 0, 1, 2)
        dialog.xmin = QtWidgets.QDoubleSpinBox()
        dialog.layout().addWidget(QLabel('Xmin'), 1, 0)
        dialog.layout().addWidget(dialog.xmin, 1, 1)
        dialog.xmax = QtWidgets.QDoubleSpinBox()
        dialog.layout().addWidget(QLabel('Xmax'), 2, 0)
        dialog.layout().addWidget(dialog.xmax, 2, 1)
        dialog.ymin = QtWidgets.QDoubleSpinBox()
        dialog.layout().addWidget(QLabel('Ymin'), 3, 0)
        dialog.layout().addWidget(dialog.ymin, 3, 1)
        dialog.ymax = QtWidgets.QDoubleSpinBox()
        dialog.layout().addWidget(QLabel('Ymax'), 4, 0)
        dialog.layout().addWidget(dialog.ymax, 4, 1)
        dialog.btn_submit = QtWidgets.QPushButton('Save')
        dialog.btn_submit.clicked.connect(dialog.accept)
        dialog.layout().addWidget(dialog.btn_submit, 5, 0, 1, 2)

        if not(dialog.exec_() == QtWidgets.QDialog.Accepted):
            raise Exception('No scale for limits set.')

        scale_dset[idx,:] = [dialog.xmin.value(), dialog.xmax.value(), dialog.ymin.value(), dialog.ymax.value()]

        gv.w.viewer.slider.setEnabled(True)
        gb_calib.le_axes_status.setText('')
        update_axes_table()
        return

    gb_calib.le_axes_status.setText('Set axis {}'.format(axes_order[pidx + 1]))


def update_axes_table():
    global gb_calib

    gv.w.gb_calib.axes_table.clear()
    gv.w.gb_calib.axes_table.setColumnCount(0)
    gv.w.gb_calib.axes_table.setRowCount(0)
    if not('axis_calibration_limits' in gv.f) or not('axis_calibration_indices' in gv.f):
        return

    findices = gv.f['axis_calibration_indices']
    limits = gv.f['axis_calibration_limits']
    scale = gv.f['axis_calibration_scale']
    labels = limits.attrs['axis_limit_order']

    ### Set table props
    gv.w.gb_calib.axes_table.setRowCount(limits.shape[1] + 1)
    gv.w.gb_calib.axes_table.setVerticalHeaderLabels(['Frame #', *labels])
    gv.w.gb_calib.axes_table.setColumnCount(findices.shape[0])
    gv.w.gb_calib.axes_table.setHorizontalHeaderLabels([str(i[0]) for i in findices])

    ### Set table data
    for i in range(findices.shape[0]):
        gv.w.gb_calib.axes_table.setItem(0, i, QtWidgets.QTableWidgetItem(str(findices[i, 0])))
        gv.w.gb_calib.axes_table.item(0, i)
        for j in range(len(labels)):
            gv.w.gb_calib.axes_table.setItem(j + 1, i, QtWidgets.QTableWidgetItem('{}: {}'.format(limits[i, j, :], scale[i, j])))


def clear_axes_calibration():
    global axes_markers
    if not('axis_calibration_indices' in gv.f) or not('axis_calibration_limits'):
        return

    answer = QtWidgets.QMessageBox.question(gv.w, 'Clear calibration', 'Are you sure you want to delete all previous calibrations?',
                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)

    if not(answer == QtWidgets.QMessageBox.Yes):
        return

    del gv.f['axis_calibration_indices']
    del gv.f['axis_calibration_limits']

    update_axes_table()

def update_axes_marker():
    global axes_markers

    x_color = (255, 0, 0)
    y_color = (0, 0, 255)
    ### Create markers
    return
    if not(bool(axes_markers)):
        axes_markers['xlims'] = pg.PlotDataItem(x=[], y=[], name='xlims',
                                                symbol='+', symbolBrush=(*x_color,255,), symbolPen=None, symbolSize=20,
                                                pen=pg.mkPen((255,0,0,255), width=2))
        gv.w.viewer.view.addItem(axes_markers['xlims'])

        axes_markers['ylims'] = pg.PlotDataItem(x=[], y=[], name='ylims',
                                                symbol='+', symbolBrush=(*y_color,255,), symbolPen=None, symbolSize=20,
                                                pen=pg.mkPen((0,0,255,255), width=2))
        gv.w.viewer.view.addItem(axes_markers['ylims'])



    ### Hide markers
    axes_markers['xlims'].setSymbolBrush((*x_color, 0))
    axes_markers['xlims'].setPen((*x_color, 0))
    axes_markers['ylims'].setSymbolBrush((*y_color, 0))
    axes_markers['ylims'].setPen((*y_color, 0))

    if not('axis_calibration_indices' in gv.f) or not('axis_calibration_limits'):
        return

    ### Check if calibration for frame exists (either for this frame or an earlier one -> lower index)
    fidx = gv.w.viewer.slider.value()
    fidx_dset = gv.f['axis_calibration_indices']
    if not(np.any(fidx_dset[:,0] <= fidx)):
        return

    ### Get current index
    idx = np.where(fidx_dset[:,0] <= fidx)[0][-1]

    ### Get limits
    limits = gv.f['axis_calibration_limits']
    labels = list(limits.attrs['axis_limit_order'])
    xlims = [limits[idx,labels.index('xmin'),:], limits[idx,labels.index('xmax'),:]]
    ylims = [limits[idx,labels.index('ymin'),:], limits[idx,labels.index('ymax'),:]]

    ### Show markers
    axes_markers['xlims'].setSymbolBrush((*x_color, 255))
    axes_markers['xlims'].setPen((*x_color, 255))
    axes_markers['ylims'].setSymbolBrush((*y_color, 255))
    axes_markers['ylims'].setPen((*y_color, 255))

    ### Set marker coordinates
    axes_markers['xlims'].setData(x=[xlims[0][0], xlims[1][0]], y=[xlims[0][1], xlims[1][1]])
    axes_markers['ylims'].setData(x=[ylims[0][0], ylims[1][0]], y=[ylims[0][1], ylims[1][1]])




