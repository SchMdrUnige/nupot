
""".. moduleauthor:: Sacha Medaer"""

import copy

import numpy as np
import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.integrals.abstract_integralB import AbstractIntegralB


# Exceptions
class IntegralBBInputError(Exception):
    pass


class FunctionBB(sp.Function):
    # Sympy advises not to create such function, instead use python
    # functions, the exception is made here to follow the Function
    # sympy logic in the integral.
    @classmethod
    def eval(cls, r, n, alpha):
        if (r.is_number and r.is_infinite):

            return sp.Rational(0)

    def doit(self, deep=False, **hints):
        r, n, alpha = self.args

        return (r**((2*n)+1)) * sp.besselk(0, alpha*r)


class IntegralBB(AbstractIntegralB):
    r"""

    Notes
    -----
    Represents the following integral:

    .. math::  \int r^{2n+1} K_0(\alpha r) d r

    """
    # ==================================================================
    @classmethod
    def _get_integrand(cls, r, *args):

        return FunctionBB(r, *args)
    # ==================================================================
    @staticmethod
    def solve_integral(r, n, alpha):
        r"""
        Notes
        -----
        Symbolic computation :

        .. math::

        \begin{split}
            \iff \int r^{2n+1} \Big(K_0(\alpha r)
            \boldsymbol{L}_{-1}(\alpha r)+K_1(\alpha r)
            \boldsymbol{L}_0(\alpha r)\Big) d r
            &=\frac{r^{2n+2}}{(2n+1)} \Big(K_0(\alpha r)
            \boldsymbol{L}_{-1}(\alpha r)
            +K_1(\alpha r)\boldsymbol{L}_0(\alpha r)\Big)\\
            &\qquad - \frac{2}{\pi(2n+1)}\int r^{2n+1} K_0(\alpha r)d r
        \end{split}

        """
        # the argument must be expanded for the factorization logic
        b_arg = (alpha * r).expand()
        k0 = sp.besselk(0, b_arg)
        k1 = sp.besselk(1, b_arg)

        if (not n):

            return (-r / alpha * k1)

        else:
            integral_ = IntegralBB.solve_integral(r, n-1, alpha)

            return ((-(r**sp.Rational((2*n)+1))/alpha*k1)
                    + (-2*n*(r**sp.Rational(2*n))/(alpha**sp.Rational(2))*k0)
                    + (4*(n**sp.Rational(2))/(alpha**sp.Rational(2))*integral_)
                   )
