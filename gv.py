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
from h5py import Dataset, File
from matplotlib import cm
from PyQt5.QtWidgets import QApplication
from numpy import ndarray
from trackpy import PandasHDFStoreBig

from gui import MainWindow, Statusbar

### Key definitions
KEY_PROCESSED: str = 'processed'
KEY_ORIGINAL: str = 'original'
KEY_TIME: str = 'time'
KEY_ATTRROT: str = 'rotation'
KEY_ATTR_FPS: str = 'fps'

KEY_ATTR_ROI_POS: str = 'rect_roi_pos'
KEY_ATTR_ROI_SIZE: str = 'rect_roi_size'

EXT_TRACKPY: str = 'trackpy'
EXT_ID_MAP: str = 'idmap'

CMAP_COLORS = cm.get_cmap('tab20').colors

app: QApplication = None
w: MainWindow = None
statusbar: Statusbar = None

filepath: str = None
open_dir: str = None
h5f: File = None
tpf: PandasHDFStoreBig = None
dset: Dataset = None
cur_obj_name: str = None
set_axes: bool = False
cmap_lut: ndarray = None
