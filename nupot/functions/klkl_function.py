
""".. moduleauthor:: Sacha Medaer"""

from typing import Union
import copy

import numpy as np
import sympy as sp
from scipy.special import kn, modstruve

import nupot.utils.constants as cst
import nupot.utils.utilities as util


class KLKLFunction(sp.Function):
    r"""

    Notes
    -----
    Represents the following function:

    .. math::  f(x) = K_0(x)\boldsymbol{L}_{-1}(x)
                      + K_1(x)\boldsymbol{L}_{0}(x)

    """
    # ==================================================================
    @property
    def argument(self):
        """ The argument of the function. """

        return self.args[0]
    # ==================================================================
    @classmethod
    def eval(cls, x):
        if (x.is_zero):

            return sp.core.S.Infinity
        if (x.is_infinite):

            return sp.core.S.Zero
    # ==================================================================
    def _eval_evalf(self, prec) -> sp.Float:
        """Return the custom evaluation of the symbol."""
        x_var = self.argument
        # evalf return a sympy float, need to convert to python float first
        x_num = float(x_var.evalf())
        K_1: float = kn(1, x_num)
        K_0: float = kn(0, x_num)
        L_m1 = modstruve(-1, x_num)
        L_0 = modstruve(0, x_num)

        return sp.core.numbers.Float((K_0*L_m1) + (K_1*L_0))
    # ==================================================================
    def _latex(self, printer) -> str:
        x_var = self.argument
        x_print = printer._print(x_var)

        return (r"\left(K_0({})".format(x_print)
                + r"\boldsymbol{L}_{-1}" + r"({})".format(x_print)
                + r" + K_1({})".format(x_print)
                + r"\boldsymbol{L}_{0}" + r"({})\right)".format(x_print))
