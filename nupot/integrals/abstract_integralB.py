
""".. moduleauthor:: Sacha Medaer"""

from abc import ABCMeta
from typing import Union

import numpy as np
import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.integrals.abstract_integral import AbstractIntegral


# Exceptions
class AbstractIntegralBInputError(Exception):
    pass


class AbstractIntegralB(AbstractIntegral):
    # ==================================================================
    def doit(self, deep=False, **hints):
        # Extracting arguments
        r, n, alpha = self.function.args
        limits = self.limits[0]
        if (len(limits) > 1):
            definite_integral = True
            if (len(limits) == 2):
                _, a = limits
                is_fully_definite = False
            elif (len(limits) == 3):
                _, a, b = limits
                is_fully_definite = True
            else:

                raise AbstractIntegralBInputError()
        else:
            definite_integral = False
        # Recursively call doit with the deep parameter
        if (deep):
            n = n.doit(deep=deep, **hints)
            alpha = alpha.doit(deep=deep, **hints)
            r = r.doit(deep=deep, **hints)

        if (definite_integral):
            if (is_fully_definite):
                if (b.is_infinite):

                    return -1*self.solve_integral(a, n, alpha)
                else:

                    return (self.solve_integral(b, n, alpha)
                            - self.solve_integral(a, n, alpha))
            else:
                if (a.is_infinite):

                    return sp.Rational(0)
                else:

                    return self.solve_integral(a, n, alpha)
        else:

            return self.solve_integral(r, n, alpha)
    # ==================================================================
    @staticmethod
    def solve_integral(r, *args):

        return NotImplementedError()
