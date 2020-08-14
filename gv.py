"""
https://github.com/thladnik/Generic3DAnnotator/gv.py - Definitions of global variables.
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
KEY_PART_CENTR_MATCH_TO_OBJ : str = 'particle_matched_to_obj'

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