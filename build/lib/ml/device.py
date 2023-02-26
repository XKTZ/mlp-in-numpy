import numpy as np
from types import ModuleType

device = np

default_cont_type = float


def set_device(dev: ModuleType):
    global device
    global default_cont_type

    device = dev
    default_cont_type = float


def set_default_continuous_type(tp):
    global default_cont_type

    default_cont_type = tp


def get_device():
    global device

    return device


def get_default_type():
    return default_cont_type
