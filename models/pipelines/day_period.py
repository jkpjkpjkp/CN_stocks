import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np
import duckdb
from pathlib import Path
from typing import Tuple, Dict
from dataclasses import dataclass
import os

from ..prelude.model import dummyLightning, dummyConfig, TM
