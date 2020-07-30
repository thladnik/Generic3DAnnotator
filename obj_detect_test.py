import h5py
import pyqtgraph as pg
from PyQt5 import QtWidgets
import cv2
import numpy as np
from matplotlib import cm
from scipy import spatial

from IPython import embed

################################
### Setup colormap for markers

colormap = cm.get_cmap("tab20")
colormap._init()
cmap_lut = np.array((colormap._lut * 255))
cmap_lut = np.append(cmap_lut[::2, :], cmap_lut[1::2, :], axis=0)

f = h5py.File('testdata/Druckabfall.hdf5', 'r')

app = QtWidgets.QApplication([])
win = QtWidgets.QMainWindow()

orig = f['original']#[30:1030, :, :, :]
#embed()
#orig.max = lambda: return

max_obj_count = 20
t, x, y, c = orig.shape

thresh = []

objects = np.nan * np.ones((t, max_obj_count, 2))
last_pos = list()

for i in range(t):
    _, im = cv2.threshold(src=orig[i,:,:,:], thresh=120, maxval=255, type=cv2.THRESH_BINARY_INV)
    #im = orig[i,:,:,:]
    thresh.append(im)


    cnts, _ = cv2.findContours(image=im, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

    mcnts = np.ones((len(cnts), 2))
    for j, cnt in enumerate(cnts):
        mcnts[j,:] = np.array([cnt[:, 0, 1].mean(), cnt[:, 0, 0].mean()]).reshape((1,2))

    if mcnts.shape[0] <= max_obj_count:
        objects[i, :mcnts.shape[0], :] = mcnts
    else:
        print('WARN: exceed max obj count')
        objects[i, :max_obj_count, :] = mcnts[:max_obj_count, :]



if False:
    # Find closest last position
    closest = (0, np.inf)
    for k in range(positions.shape[1]):
        validpos = np.logical_not(np.isnan(positions[:, k, 0]))

        if np.all(np.logical_not(validpos)) and j == k:
            closest = (k, 0)
            break
        tdist = np.where(validpos)
        tdist = tdist[0][-1]

        pos = positions[validpos, k, :][-1, :]
        dist = (i - tdist) * np.sqrt(np.square(pos[0]-mcnt[0]) + np.square(pos[1]-mcnt[1]))
        #embed()
        if dist < closest[1]:
            closest = (k, dist)

    positions[i,closest[0],:] = mcnt

thresh = np.array(thresh)

plotItems = []
def setupMarkers():
    global plotItems, imv2, cmap_lut

    for i in range(cmap_lut.shape[0]):
        print('Set marker ', i)
        plotItems.append(pg.PlotDataItem(x=[0], y=[0],
                                   symbolPen=(*cmap_lut[i,:3], 0,), symbolBrush=None, symbol='o', penSize=15,
                                   name='object{}'.format(i)))

        imv2.addItem(plotItems[i])


def updateMarkers():
    global plotItems, objects, imv2

    Ms = positions[imv2.currentIndex]
    ### Get brush
    for i in range(len(plotItems)):

        sym_pen = plotItems[i].opts['symbolPen']
        if isinstance(sym_pen, tuple):
            rgb = sym_pen[:3]
        else:
            rgb = sym_pen.color().getRgb()[:3]

        if len(Ms) > i:
            M = Ms[i]
            plotItems[i].setData(x=[M[0]], y=[M[1]])
            #print('Set {} to {}/{}'.format(i, M[0], M[1]))
            rgba = (*rgb, 255)
        else:
            rgba = (*rgb, 0)

        plotItems[i].setSymbolPen(*rgba)

win.w = QtWidgets.QWidget()
win.w.setLayout(QtWidgets.QHBoxLayout())
win.setCentralWidget(win.w)


imv1 = pg.ImageView()
imv1.setImage(orig)
win.w.layout().addWidget(imv1)

imv2 = pg.ImageView()
imv2.setImage(thresh)
win.w.layout().addWidget(imv2)

setupMarkers()
imv2.sigTimeChanged.connect(updateMarkers)
imv2.sigTimeChanged.connect(lambda: imv1.setCurrentIndex(imv2.currentIndex))
imv1.sigTimeChanged.connect(lambda: imv2.setCurrentIndex(imv1.currentIndex))

win.resize(1400,900)
win.show()
app.exec_()