
""".. moduleauthor:: Sacha Medaer"""

import copy

import numpy as np
import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.integrals.abstract_integral import AbstractIntegral
from nupot.functions.klkl_function import KLKLFunction


# Exceptions
class IntegralAAInputError(Exception):
    pass


class FunctionAA(sp.Function):
    # Sympy advises not to create such function, instead use python
    # functions, the exception is made here to follow the Function
    # sympy logic in the integral.
    def doit(self, deep=False, **hints):
        x, rho, mu, a, r = self.args

        return (x**rho) * ((x-a)**mu) * sp.exp(-r*(x**sp.Rational(1, 2)))


class IntegralAA(AbstractIntegral):
    r"""

    Notes
    -----
    Represents the following integral:

    .. math::  \int_{a}^{\infty} x^{\rho} (x-a)^{\mu}
               e^{-r \sqrt{x}} dx

    """
    # ==================================================================
    @classmethod
    def _get_integrand(cls, x, *args):

        return FunctionAA(x, *args)
    # ==================================================================
    def doit(self, deep=False, **hints):
        # Extracting arguments
        args = self.function.args
        x, rho, mu, a, r = args
        limits = self.limits[0]
        if (len(limits) > 1):
            definite_integral = True
            if (len(limits) == 2):
                _, a_ = limits
                is_fully_definite = False
            elif (len(limits) == 3):
                _, a_, b = limits
                is_fully_definite = True
            else:

                raise AbstractIntegralBInputError()
        else:
            definite_integral = False
        # Recursively call doit with the deep parameter
        args_ = []
        for i in range(len(args)):
            args_.append(args[i].doit(deep=deep, **hints))

        if (not (a.equals(a_))):  # definition of this integral

            return None

        if (definite_integral):
            if (is_fully_definite):
                if (isinstance(b, sp.core.numbers.Infinity)):

                    return self.solve_def_integral(a, *args_[1:])
                else:

                    return (self.solve_def_integral(a, *args_[1:])
                            - self.solve_def_integral(b, *args_[1:]))
        # all other possibilities are not defined
    # ==================================================================
    @staticmethod
    def solve_def_integral(x, rho, mu, a, r):
        r"""
        Notes
        -----
        sp.Symbolic computation :

        .. math::

        \begin{cases}
            \text{if } \rho=-\frac{1}{2}\,\wedge\,\mu>0\text{ : }
            &\int_{a}^{\infty} \ldots dx
            = \sp.gamma(\mu+1) 2^{\mu+\frac{3}{2}}
            \pi^{-\frac{1}{2}} r^{-\frac{1}{2}-\mu} a^{\frac{1}{2}\mu
            + \frac{1}{4}} K_{\mu+\frac{1}{2}}(a^{\frac{1}{2}} r)\\
            \text{if } \rho=0\,\wedge\,\mu>0\text{ : }&\int_{a}^{\infty}
            \ldots dx
            = \sp.gamma(\mu+1) 2^{\mu+\frac{3}{2}} \pi^{-\frac{1}{2}}
            r^{-\frac{1}{2}-\mu} a^{\frac{1}{2}\mu + \frac{3}{4}}
            K_{\mu+\frac{3}{2}}(a^{\frac{1}{2}} r)\\
            \text{if } \rho=-1\,\wedge\,\mu=\frac{1}{2}\text{ : }
            &\int_{a}^{\infty} \ldots dx = 2a^{\frac{1}{2}}
            K_1(a^{\frac{1}{2}} r)  - \pi a^{\frac{1}{2}} \\
            &\qquad \qquad \qquad + \pi a r \Big(K_0(a^{\frac{1}{2}} r)
            \boldsp.Symbol{L}_{-1}(a^{\frac{1}{2}} r)
            +K_1(a^{\frac{1}{2}} r)
            \boldsp.Symbol{L}_0(a^{\frac{1}{2}} r)\Big)\\
            \text{if } \rho=-1\,\wedge\,\mu=-\frac{1}{2}\text{ : }
            &\int_{a}^{\infty} \ldots dx =  \pi a^{-\frac{1}{2}} -r\pi
            \Big(K_0(a^{\frac{1}{2}} r)
            \boldsp.Symbol{L}_{-1}(a^{\frac{1}{2}} r)
            +K_1(a^{\frac{1}{2}} r)
            \boldsp.Symbol{L}_0(a^{\frac{1}{2}} r)\Big)\\
            \text{if } \rho=-1\,\wedge\,\mu=-\frac{3}{2}\text{ : }
            &\int_{a}^{\infty} \ldots dx
            = - 2 a^{-1} r K_0(a^{\frac{1}{2}} r)
            - \pi a^{-\frac{3}{2}}\\
            &\qquad \qquad \qquad
            + \pi a^{-1} r\Big(K_0(a^{\frac{1}{2}} r)
            \boldsp.Symbol{L}_{-1}(a^{\frac{1}{2}} r)
            +K_1(a^{\frac{1}{2}} r)
            \boldsp.Symbol{L}_0(a^{\frac{1}{2}} r)\Big)\\
             \text{if } \rho=-\frac{3}{2}\,\wedge\,\mu
             =-\frac{3}{2}\text{ : }
             &\int_{a}^{\infty} \ldots dx = - 4r a^{-\frac{3}{2}}
             K_1(a^{\frac{1}{2}} r) + \pi a^{-\frac{3}{2}} r\\
             &\qquad \qquad \qquad - \frac{\pi r^2}{a}
             \Big(K_0(a^{\frac{1}{2}} r)
             \boldsp.Symbol{L}_{-1}(a^{\frac{1}{2}} r)
             +K_1(a^{\frac{1}{2}} r)
             \boldsp.Symbol{L}_0(a^{\frac{1}{2}} r)\Big)
        \end{cases}

        """
        # the argument must be expanded for the factorization logic
        b_arg = (r*(a**sp.Rational(1, 2))).expand()
        klkl = KLKLFunction(b_arg)
        if ((rho == sp.Rational(-1, 2))):  # Integral Table
            bessel_order = mu+0.5
            bessel_fct = sp.besselk(sp.Rational(bessel_order), b_arg)
            if (bessel_order > 1):    # will transform K_n in K_0 and K_1
                bessel_fct = sp.simplify(bessel_fct)

            return (sp.gamma(mu+1)*sp.Rational(2**(mu+1.5))
                    *(sp.pi**sp.Rational(-1, 2))
                    *(r**sp.Rational(-0.5-mu))*(a**sp.Rational(0.5*mu+0.25))
                    *bessel_fct)

        elif (not rho): # Integral Table
            bessel_order = mu+1.5
            bessel_fct = sp.besselk(sp.Rational(bessel_order), b_arg)
            if (bessel_order > 1):    # will transform K_n in K_0 and K_1
                bessel_fct = sp.simplify(bessel_fct)

            return (sp.gamma(mu+1)*sp.Rational(2**(mu+1.5))
                    *(sp.pi**sp.Rational(-1, 2))
                    *(r**sp.Rational(-0.5-mu))*(a**sp.Rational(0.5*mu+0.75))
                    *bessel_fct)

        elif ((rho == sp.Rational(-1)) and (mu == sp.Rational(1, 2))):  # Feynman

            return ((2*(a**sp.Rational(1, 2))*sp.besselk(1, b_arg))
                    + (sp.pi*a*r*klkl) - (sp.pi*(a**sp.Rational(1, 2))))

        elif ((rho == sp.Rational(-1)) and (mu == sp.Rational(-1, 2))): # Feynman

            return (-r*sp.pi*klkl) + (sp.pi*(a**sp.Rational(-1, 2)))

        elif ((rho == sp.Rational(-1)) and (mu == sp.Rational(-3, 2))): # Feynman

            return ((sp.pi*(a**sp.Rational(-1))*r*klkl)
                    - (sp.pi*(a**sp.Rational(-3, 2)))
                    - (2*(a**sp.Rational(-1))*r*sp.besselk(0, b_arg)))

        elif ((rho == sp.Rational(-3, 2)) and (mu == sp.Rational(-3, 2))): # Feynman

            return ((sp.pi*(a**sp.Rational(-3, 2))*r)
                    - (sp.pi*(a**sp.Rational(-1))*(r**2)*klkl)
                    - (4*(a**sp.Rational(-3, 2))*r*sp.besselk(1, b_arg)))

        else:       # Integration by parts
            # Tried mu < 3/2, integrate by part with integ (t-a)
            # but then loop for ever
            if (rho > 0):
                part_factor = sp.Rational(-1)
                coeff_first_integ = sp.Rational(1, mu+1)
                coeff_first_deriv = sp.Rational(-1, 2)*r
                coeff_second_deriv = copy.copy(rho)
                rho_int_1 = rho + sp.Rational(-1, 2)
                mu_int_1 = mu + sp.Rational(1)
                integral_1 = IntegralAA.solve_def_integral(x, rho_int_1,
                                                           mu_int_1, a, r)
                rho_int_2 = rho + sp.Rational(-1)
                mu_int_2 = mu + sp.Rational(1)
                integral_2 = IntegralAA.solve_def_integral(x, rho_int_2,
                                                           mu_int_2, a, r)

                return ((part_factor*coeff_first_integ*coeff_first_deriv
                         *integral_1)
                        + (part_factor*coeff_first_integ*coeff_second_deriv
                           *integral_2)
                        )
            else:
                part_factor = sp.Rational(-1)
                coeff_first_integ = sp.Rational(1, rho+1)
                coeff_first_deriv = sp.Rational(-1, 2)*r
                coeff_second_deriv = copy.copy(mu)
                rho_int_1 = rho + sp.Rational(1) + sp.Rational(-1, 2)
                mu_int_1 = copy.copy(mu)
                integral_1 = IntegralAA.solve_def_integral(x, rho_int_1,
                                                           mu_int_1, a, r)
                rho_int_2 = rho + sp.Rational(1)
                mu_int_2 = mu - sp.Rational(1)
                integral_2 = IntegralAA.solve_def_integral(x, rho_int_2,
                                                           mu_int_2, a, r)

                return ((part_factor*coeff_first_integ*coeff_first_deriv
                         *integral_1)
                        + (part_factor*coeff_first_integ*coeff_second_deriv
                           *integral_2)
                        )
