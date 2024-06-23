
""".. moduleauthor:: Sacha Medaer"""

import copy

import numpy as np
import sympy as sp
from scipy.special import digamma, factorial

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.integrals.abstract_integral import AbstractIntegral


# Exceptions
class KSiIntegralInputError(Exception):
    pass


class KSiFunction(sp.Function):
    # Sympy advises not to create such function, instead use python
    # functions, the exception is made here to follow the Function
    # sympy logic in the integral.
    def doit(self, deep=False, **hints):
        r, alpha = self.args
        b_arg = (alpha * r).expand()
        k0 = sp.besselk(0, b_arg)

        return k0 / b_arg


class KSiIntegral(AbstractIntegral):
    r"""

    Notes
    -----
    Represents the following function:

    .. math::  f(x) = \left.KSi(\alpha r)\right|_a^b
                    = \left.KSi(\alpha r)\right|_a^\infty
                        - \left.KSi(\alpha r)\right|_b^\infty

    """
    # ==================================================================
    @classmethod
    def _get_integrand(cls, r, *args):

        return KSiFunction(r, *args)
    # ==================================================================
    def doit(self, deep=False, **hints):
        # Extracting arguments
        args = self.function.args
        if (len(args) == 2):
            r, alpha = args
        elif (len(args) == 3):
            r, alpha, n = args
        else:

            raise KSiIntegralInputError()
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
            alpha = alpha.doit(deep=deep, **hints)
            r = r.doit(deep=deep, **hints)

        if (definite_integral):
            if (is_fully_definite):
                if (isinstance(b, sp.core.numbers.Infinity)):

                    return self.solve_def_integral(a, alpha, *args[2:])
                else:

                    return (self.solve_def_integral(a, alpha, *args[2:])
                            - self.solve_def_integral(b, alpha, *args[2:]))
        # all other possibilities are not defined
    # ==================================================================
    @staticmethod
    def solve_def_integral(r, alpha, n=6):
        r"""Return the custom evaluation of the definite integral
        with limits between r and infinite.

        Notes
        -----
        Represents the following series expansion:

        .. math::  \left.KSi(\alpha r)\right|_a^\infty
                    = \int_a^\infty \frac{K_0(\alpha r)}{\alpha r} d r
                    = \frac{1}{\alpha} \int_{\alpha a}^\infty
                        \frac{K_0(r')}{r'} d r'
                    = \frac{1}{\alpha}\left(\frac{\pi^2}{24}
                        + \frac{1}{2}\left(\ln\left(\frac{\alpha a}{2}
                          \right) + \gamma\right)^2 - \sum_{k=1}^\infty
                          \left(\psi\left(k+1\right) + \frac{1}{2k}
                          -\ln\left(\frac{\alpha a}{2}\right)\right)
                          \frac{\left(\frac{\alpha a}{2}\right)^{2k}}
                               {2k(k!)^2}\right)

        """
        log_num = sp.log(alpha*r/2.0)
        res = (np.pi**2) / 24.0
        res += 0.5 * (log_num + np.euler_gamma)**2
        for k in range(1, n+1):
            res -= ((digamma(k+1) + (1/(2*k)) - log_num)
                    *(((alpha*r/2.0)**2) / (2*k*(factorial(k)**2)))
                    )
        res /= alpha

        return res
