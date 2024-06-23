
""".. moduleauthor:: Sacha Medaer"""

import copy

import numpy as np
import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.integrals.abstract_integralB import AbstractIntegralB
from nupot.integrals.ksi_integral import KSiIntegral


# Exceptions
class IntegralBEInputError(Exception):
    pass


class FunctionBE(sp.Function):
    # Sympy advises not to create such function, instead use python
    # functions, the exception is made here to follow the Function
    # sympy logic in the integral.
    @classmethod
    def eval(cls, r, n, alpha):
        if (r.is_number and r.is_infinite):

            return sp.Rational(0)

    def doit(self, deep=False, **hints):
        r, n, alpha = self.args

        return (r**((-2*n)-1)) * sp.besselk(0, alpha*r)


class IntegralBE(AbstractIntegralB):
    r"""

    Notes
    -----
    Represents the following integral:

    .. math::  \int r^{-2n-1} K_0(\alpha r)d r

    """
    # ==================================================================
    @classmethod
    def _get_integrand(cls, r, *args):

        return FunctionBE(r, *args)
    # ==================================================================
    @staticmethod
    def solve_integral(r, n, alpha):
        r"""
        Notes
        -----
        Symbolic computation :

        .. math:: \int r^{-2n-1} K_0(\alpha  r) d r
                  = -\frac{1}{2n}r^{-2n} K_0(\alpha  r)
                    + \frac{\alpha }{4n^2} r^{-2n+1}K_1(\alpha  r)
                    + \frac{a}{4n^2}\int r^{-2n+1} K_0(\alpha  r) d r

        """
        # the argument must be expanded for the factorization logic
        b_arg = (alpha * r).expand()
        k0 = sp.besselk(0, b_arg)
        k1 = sp.besselk(1, b_arg)

        if (not n):

            return alpha * -1 * KSiIntegral.solve_def_integral(r, alpha)

        else:
            integral_ = IntegralBE.solve_integral(r, n-1, alpha)

            return ((sp.Rational(-1, 2*n)*(r**(-2*n))*k0)
                    + (sp.Rational(1, 4*(n**2))*alpha*(r**((-2*n)+1))*k1)
                    - (sp.Rational(1, 4*(n**2)) * (alpha**2) * integral_)
                   )
