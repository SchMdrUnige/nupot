
""".. moduleauthor:: Sacha Medaer"""

from abc import ABCMeta
from typing import Union

import numpy as np
import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util


class AbstractIntegral(sp.Integral):
    # ==================================================================
    def __new__(cls, *args, **kwargs):
        """Instead of the first argument being a function, the first
        argument of the custom integral is a tuple containing the
        parameters of the function representing the integrand of a
        given integral.
        """
        if (isinstance(args[0], tuple)): # First call with custom definition
            integrand = cls._get_integrand(args[1][0], *args[0])
            if (len(args[1]) > 1):

                return super().__new__(cls, integrand, args[1], **kwargs)
            else:

                return super().__new__(cls, integrand, args[1][0], **kwargs)
        else:   # call from super w/ args (func, limits) or (func,)

            return super().__new__(cls, *args, **kwargs)
    # ==================================================================
    @classmethod
    def _get_integrand(cls, *args):

        return NotImplementedError()
