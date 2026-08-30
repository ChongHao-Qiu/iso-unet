"""ISO-UNet package.

Models for predicting d18Op from climate fields with explicit
region-aware routing and climate-state conditioning.
"""
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version('iso_unet')
except PackageNotFoundError:
    __version__ = '0.0.0'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from . import utils
from .data import Dataset, SequenceDataset
from .model import FNO2d, LinearReg2d, LinearReg2dPerGrid, DisentangledTransformer
from .convlstm import ConvLSTM
from .trainer import Trainer
