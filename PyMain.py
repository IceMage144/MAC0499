# This script was made to load all python libraries
# at the beggining of the game, so the game will not
# be interupted for this

from random import choice, random
import math
import time
import base64

from godot import exposed, export
from godot.bindings import *
from godot.globals import *

import torch
import numpy
import matplotlib
import pickle

import util
import AIs.QLAI
import AIs.structs

@exposed
class PyMain(Node):
    def _ready(self):
        pass
