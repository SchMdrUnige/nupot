
""".. moduleauthor:: Sacha Medaer"""

from abc import ABCMeta
from typing import Union

import numpy as np
import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util


class AbstractParameter(sp.Function):
    r"""Represent a parameter which can be either constant or depends
    on other parameters or variables.
    """

    def _eval_evalf(self, prec):

        return NotImplementedError
