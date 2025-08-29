#-*- coding:utf-8 -*-
from enum import IntEnum, Enum

class LabelEnum(IntEnum):
    BACKGROUND = 0
    LUNG = 1
    Nodule = 2

class LabelEnum_Plus(IntEnum):
    BACKGROUND = 0
    Nodule = 1

class FilterMethods(Enum):
    CUBIC = "CUBIC"
    LANCZOS2 = "LANCZOS2"
    LANCZOS3 = "LANCZOS3"
    BOX = "BOX"
    LINEAR = "LINEAR"
