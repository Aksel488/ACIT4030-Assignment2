import numpy as np
import os
import trimesh
from matplotlib import pyplot as plt
from utils import read_off, show_mesh, download_modelnet10

import tensorflow as tf
from tensorflow import keras


# Download dataset if not exists
path_here = os.path.dirname(__file__)
DATA_DIR = os.path.join(path_here, 'datasets/ModelNet10')
if not os.path.isdir(DATA_DIR):
    print("Downloading")
    DATA_DIR = download_modelnet10()

path = 'bathtub/train/bathtub_0001.off'
show_mesh(os.path.join(DATA_DIR, path))



