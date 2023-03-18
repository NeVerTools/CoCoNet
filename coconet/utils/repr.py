"""
Module repr.py

This module contains utility methods for the representation of objects

Author: Andrea Gimelli, Giacomo Rosato, Stefano Demarchi

"""
import json

from coconet import RES_DIR

JSON_PATH = RES_DIR + '/json'


def read_json_data() -> tuple:
    with open(JSON_PATH + '/blocks.json', 'r') as fdata:
        block_data = json.load(fdata)

    with open(JSON_PATH + '/properties.json', 'r') as fdata:
        prop_data = json.load(fdata)

    with open(JSON_PATH + '/functionals.json', 'r') as fdata:
        func_data = json.load(fdata)

    return block_data, prop_data, func_data
