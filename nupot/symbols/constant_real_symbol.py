
""".. moduleauthor:: Sacha Medaer"""

import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util


class ConstantRealSymbol(sp.Symbol):
    r"""Represent a symbol with constant real value which is passed as
    argument in the constructor.
    """
    def __new__(cls, value: float, *args, **kwargs):
        # Ignore the additional argument fort this child class and call
        # the parent constructor

        return super().__new__(cls, *args, **kwargs, real=True)
    # ==================================================================
    def __init__(self, value: float, *args, **kwargs):
        # N.B.: not constructor defined for the Function object in sympy
        # calling the parent init will crash as the str name of the
        # function is passed
        self._value: float = value

        return None
    # ==================================================================
    def _eval_evalf(self, prec):

        return sp.Float(self._value, prec)


if __name__ == '__main__':

    an_value = 12.0
    an = ConstantRealSymbol(an_value, 'Z')
    print(an.evalf())
