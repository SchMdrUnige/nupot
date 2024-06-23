
""".. moduleauthor:: Sacha Medaer"""

import copy

import numpy as np
import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.integrals.abstract_integralB import AbstractIntegralB


# Exceptions
class IntegralBCInputError(Exception):
    pass


class FunctionBC(sp.Function):
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

        return (r**(2*n)) * k1


class IntegralBC(AbstractIntegralB):
    r"""

    Notes
    -----
    Represents the following integral:

    .. math::  \int r^{2n} K_1(\alpha r) d r

    """
    # ==================================================================
    @classmethod
    def _get_integrand(cls, r, *args):

        return FunctionBC(r, *args)
    # ==================================================================
    @staticmethod
    def solve_integral(r, n, alpha):
        r"""
        Notes
        -----
        Symbolic computation :

        .. math:: \int r^{2n} K_1(\alpha r) d r
                  = -\frac{1}{\alpha }r^{2n} K_0(\alpha r)
                    - \frac{2n}{a} r^{2n-1} K_1(\alpha r)
                    + \frac{2n(2n-2)}{a}\int r^{2n-2} K_1(\alpha r) d r

        """
        # the argument must be expanded for the factorization logic
        b_arg = (alpha * r).expand()
        k0 = sp.besselk(0, b_arg)
        k1 = sp.besselk(1, b_arg)

        if (not n):

            return (-k0 / alpha)

        else:
            integral_ = IntegralBC.solve_integral(r, n-1, alpha).doit()

            return (-(r**sp.Rational(2*n)/alpha*k0)
                    + (-2*n*(r**sp.Rational(2*n-1))/(alpha**sp.Rational(2))*k1)
                    + (2*n*((2*n)-2)/(alpha**sp.Rational(2))*integral_)
                   )
