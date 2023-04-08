"""
Module rep.py

This module contains utility methods for the representation of objects

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""

import json
import traceback

from PyQt6.QtCore import QLocale, QRegularExpression
from PyQt6.QtGui import QIntValidator, QDoubleValidator, QRegularExpressionValidator

from coconet import RES_DIR
from coconet.view.ui.dialog import MessageDialog, MessageType

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


def create_variables_from(v_name: str, v_dim: tuple) -> list:
    """
    This method creates a list of variables describing
    the tuple v_dim with the name v_name.

    Parameters
    ----------
    v_name : str
        The variable main name.
    v_dim : tuple
        The variable shape.

    Returns
    ----------
    list
        The list of string variables.

    """

    temp_list = []
    ped_list = []
    var_list = []

    # Add underscore
    v_name += '_'
    for k in v_dim:
        if len(temp_list) == 0:
            for i in range(k):
                temp_list.append(str(i))
        else:
            for i in range(k):
                for p in temp_list:
                    p = f"{p}-{i}"
                    ped_list.append(p)
            temp_list = ped_list
            ped_list = []

    for p in temp_list:
        var_list.append(f"{v_name}{p}")

    return var_list


def text2tuple(text: str) -> tuple:
    """
    This method takes a string in format "(n,m,l)..."
    converts it into a variable of type tuple with the given dimensions.

    Parameters
    ----------
    text: str
        Input string to convert.

    Returns
    ----------
    tuple
        The converted tensor set.

    """

    output_tuple = tuple()
    text = str(text)

    if len(text.split(",")) > 1:
        for token in text.replace("(", "").replace(")", "").split(","):
            if token != "":
                num = int(token)
                output_tuple += (num,)
    else:
        output_tuple += (int(text),)

    return output_tuple


def format_data(params: dict) -> dict:
    """
    This function re-formats a complete dictionary of block attributes in the format
    <key> : str
    <value> : expected type

    Parameters
    ----------
    params : dict
        The original dictionary

    Returns
    ----------
    dict
        The formatted dictionary

    """

    converted_dict = dict()
    value = None

    try:
        for param_name, param_value in params.items():
            if param_value[1] == '':
                continue
            if param_value[2] == 'Tensor':
                value = text2tuple(param_value[1])
            elif param_value[2] == 'int':
                value = int(param_value[1])
            elif param_value[2] == 'list of ints':
                value = list(map(int, param_value[1].split(', ')))
            elif param_value[2] == 'boolean':
                value = param_value[1] == 'True'
            elif param_value[2] == 'float':
                value = float(param_value[1])
            converted_dict[param_name] = value
    except Exception as e:
        dialog = MessageDialog(str(e), MessageType.ERROR)
        dialog.exec()

    return converted_dict


def dump_exception(e: Exception):
    # TODO format
    traceback.print_exc()
