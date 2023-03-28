"""
Module repr.py

This module contains utility methods for the representation of objects

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

import json

from PyQt6.QtCore import QLocale, QRegularExpression
from PyQt6.QtGui import QIntValidator, QDoubleValidator, QRegularExpressionValidator

from coconet import RES_DIR

JSON_PATH = RES_DIR + '/json'


class ArithmeticValidator:
    """
    This class collects the possible validators for
    the editing dialogs.

    INT : (QIntValidator)
        Integer validator.
    FLOAT : (QDoubleValidator)
        Floating-point validator.
    TENSOR : (QRegExpValidator)
        Tensor ("nxmxl", "nXmXl", "n,m,l") with n, m, l
        integers validator.
    TENSOR_LIST : (QRegExpValidator)
        List of Tensors.

    """

    locale = QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)

    INT = QIntValidator()
    INT.setLocale(locale)

    FLOAT = QDoubleValidator()
    FLOAT.setLocale(locale)

    TENSOR = QRegularExpressionValidator(QRegularExpression('(([0-9])+(,\s?[0-9]+)*)'))
    TENSOR.setLocale(locale)

    SAMPLE = QRegularExpressionValidator(QRegularExpression('^(?:\d+(?:\.\d*)?|\.\d+)(?:,(?:\d+(?:\.\d*)?|\.\d+))*$'))


def read_json_data() -> tuple:
    with open(JSON_PATH + '/blocks.json', 'r') as fdata:
        block_data = json.load(fdata)

    with open(JSON_PATH + '/properties.json', 'r') as fdata:
        prop_data = json.load(fdata)

    with open(JSON_PATH + '/functionals.json', 'r') as fdata:
        func_data = json.load(fdata)

    return block_data, prop_data, func_data
