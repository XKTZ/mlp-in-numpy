import numpy as np
from types import ModuleType

device = np


def set_device(dev: ModuleType):
    global device

    device = dev
