from os.path import join as path_join, sep
from sys import path

VELES_PATH = path_join("C:" + sep, "veles-stable-0.10.0")
path.append(VELES_PATH)

import veles.znicz