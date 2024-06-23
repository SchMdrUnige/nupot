
""".. moduleauthor:: Sacha Medaer"""

import copy

import numpy as np
import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.functions.klkl_function import KLKLFunction
from nupot.integrals.abstract_integralB import AbstractIntegralB
from nupot.integrals.integralBB import IntegralBB


# Exceptions
class IntegralBAInputError(Exception):
    pass


class FunctionBA(sp.Function):
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
        klkl = KLKLFunction(b_arg).evalf()

        return (r**((2*n)+1)) * klkl


class IntegralBA(AbstractIntegralB):
    r"""

    Notes
    -----
    Represents the following integral:

    .. math::  \int r^{2n+1} \Big(K_0(\alpha r)
               \boldsymbol{L}_{-1}(\alpha r) +K_1(\alpha r)
               \boldsymbol{L}_0(\alpha r)\Big) dr

    """
    # ==================================================================
    @classmethod
    def _get_integrand(cls, r, *args):

        return FunctionBA(r, *args)
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
        klkl = KLKLFunction(b_arg)
        integral_ = IntegralBB.solve_integral(r, n, alpha)

        return (((r**sp.Rational((2*n)+2)) * klkl * sp.Rational(1, (2*n)+1))
                - (sp.Rational(2, (2*n)+1) * integral_ / sp.pi)
               )
