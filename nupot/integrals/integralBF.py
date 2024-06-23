
""".. moduleauthor:: Sacha Medaer"""

import copy

import numpy as np
import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.functions.klkl_function import KLKLFunction
from nupot.integrals.abstract_integralB import AbstractIntegralB


# Exceptions
class IntegralBFInputError(Exception):
    pass


class FunctionBF(sp.Function):
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
        k0 = sp.besselk(0, b_arg)

        return (r**(2*n)) * k0


class IntegralBF(AbstractIntegralB):
    r"""

    Notes
    -----
    Represents the following integral:

    .. math::  \int r^{2n} K_0(\alpha r) d r

    """
    # ==================================================================
    @classmethod
    def _get_integrand(cls, r, *args):

        return FunctionBF(r, *args)
    # ==================================================================
    @staticmethod
    def solve_integral(r, n, alpha):
        r"""
        Notes
        -----
        Symbolic computation :

        .. math:: \int r^{2n} K_0(a^{\frac{1}{2}} r) d r
                   = -\frac{1}{a^{\frac{1}{2}}}r^{2n}
                        K_1(a^{\frac{1}{2}} r)
                     - \frac{(2n-1)}{a} r^{2n-1}
                        K_0(a^{\frac{1}{2}} r)
                     + \frac{(2n-1)^2}{a}\int r^{2n-2}
                        K_0(a^{\frac{1}{2}} r) 

        """
        # the argument must be expanded for the factorization logic
        b_arg = (alpha * r).expand()

        if (not n):
            klkl = KLKLFunction(b_arg)

            return r * sp.pi * klkl / 2.0

        else:
            k0 = sp.besselk(0, b_arg)
            k1 = sp.besselk(1, b_arg)
            integral_ = IntegralBF.solve_integral(r, n-1, alpha).doit()

            return (- ((r**sp.Rational(2*n))*k1/alpha)
                    - (((2*n)-1)*(r**sp.Rational((2*n)-1))*k0
                       /(alpha**sp.Rational(2)))
                    + ((((2*n)-1)**sp.Rational(2))*integral_
                       /(alpha**sp.Rational(2)))
                   )
