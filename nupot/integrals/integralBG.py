
""".. moduleauthor:: Sacha Medaer"""

import copy

import numpy as np
import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.functions.klkl_function import KLKLFunction
from nupot.integrals.integralBF import IntegralBF
from nupot.integrals.abstract_integralB import AbstractIntegralB


# Exceptions
class IntegralBGInputError(Exception):
    pass


class FunctionBG(sp.Function):
    # Sympy advises not to create such function, instead use python
    # functions, the exception is made here to follow the Function
    # sympy logic in the integral.
    @classmethod
    def eval(cls, r, n, alpha):
        if (r.is_number and r.is_infinite):

            return sp.Rational(0)

    def doit(self, deep=False, **hints):
        r, n, alpha = self.args
        b_arg = (alpha * r).expand()
        k1 = sp.besselk(1, b_arg)

        return (r**((2*n)+1)) * k1


class IntegralBG(AbstractIntegralB):
    r"""

    Notes
    -----
    Represents the following integral:

    .. math::  \int r^{2n+1} K_1(\alpha r) d r

    """
    # ==================================================================
    @classmethod
    def _get_integrand(cls, r, *args):

        return FunctionBG(r, *args)
    # ==================================================================
    @staticmethod
    def solve_integral(r, n, alpha):
        r"""
        Notes
        -----
        Symbolic computation :

        .. math:: \int r^{2n+1} K_1(\alpha  r) d r
                  = -\frac{1}{\alpha }r^{2n+1} K_0(\alpha  r)
                    + \frac{(2n+1)}{\alpha}\int r^{2n} K_0(\alpha r) d r

        """
        # the argument must be expanded for the factorization logic
        b_arg = (alpha * r).expand()
        k0 = sp.besselk(0, b_arg)
        integral_ = IntegralBF.solve_integral(r, n, alpha).doit()

        return (- ((r**sp.Rational((2*n)+1))*k0/alpha)
                + (((2*n)+1)*integral_/alpha)
               )
