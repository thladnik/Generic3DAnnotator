import h5py
from gui import HDF5ImageView
from PyQt5.QtWidgets import QApplication
from numpy import ndarray
from gui import MainWindow, Statusbar

### Key definitions
KEY_PROCESSED : str = 'processed'
KEY_ORIGINAL  : str = 'original'
KEY_OBJLIST   : str = 'obj_list'
KEY_OBJSTR    : str = 'obj'
KEY_NODES     : str = 'nodes'
KEY_NODEINTERP: str = 'node_interp'
KEY_ROT       : str = 'rotation'
KEY_FRAMEIDCS : str = 'frame_indices'
KEY_TIME      : str = 'time'
KEY_FPS       : str = 'fps'
KEY_PARTICLES : str = 'particles'
KEY_PART_CENTR: str = 'particle_centroids'
KEY_PART_AREA : str = 'particle_area'

app             : QApplication    = None
w               : MainWindow      = None
statusbar       : Statusbar       = None

filepath        : str             = None
open_dir        : str             = None
f               : h5py.File       = None
dset            : h5py.Dataset    = None
cur_obj_name    : str             = None
set_axes        : bool            = False
cmap_lut        : ndarray         = None