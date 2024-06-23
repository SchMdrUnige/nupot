import pytest
import math

import numpy as np
import sympy as sp
from scipy.integrate import quad

from nupot.functions.klkl_function import KLKLFunction
from nupot.integrals.integralAA import FunctionAA, IntegralAA
import nupot.utils.constants as cst
import nupot.utils.utilities as util

# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------


@pytest.mark.integral
def test_particular_integralAA_1():
    r"""Should fail if the obtained result does not correspond to
    theoretical expectation.

    Notes
    -----
    Represents the following integral:

    .. math::  int_{a}^{\infty} d t\, e^{-r \sqrt{t}}\, t^{\frac{1}{2}}
               \big(t - a\big)^{\frac{1}{2}}
               = -\frac{2a}{r^2} K_2(a^{\frac{1}{2}}r)
               + \frac{2a^{\frac{3}{2}}}{r} K_3(a^{\frac{1}{2}}r)

    """

    a, r, x = sp.symbols("a r dummyx")
    b_arg = r*(a**sp.Rational(1, 2))
    k2 = sp.besselk(2, b_arg)
    k3 = sp.besselk(3, b_arg)

    res_theo = ((-2*a*sp.simplify(k2)*(r**sp.Rational(-2)))
                + (2*(a**sp.Rational(3,2))*k3*(r**sp.Rational(-1)))
               )

    rho = sp.Rational(1, 2)
    mu = sp.Rational(1, 2)
    res = IntegralAA((rho, mu, a, r), (x, a, np.inf)).doit()

    # Tests
    assert (res.equals(res_theo))


@pytest.mark.integral
def test_particular_integralAA_2():
    r"""Should fail if the obtained result does not correspond to
    theoretical expectation.

    Notes
    -----
    Represents the following integral:

    .. math::  \int_{a}^{\infty} d t\, e^{-r \sqrt{t}}\,
               t^{-\frac{1}{2}} \big(t - a\big)^{\frac{1}{2}}
               = \frac{2 a^{\frac{1}{2}}}{r} K_1(a^{\frac{1}{2}} r)

    """

    a, r, x = sp.symbols("a r dummyx")
    b_arg = r*(a**sp.Rational(1, 2))
    k1 = sp.besselk(1, b_arg)
    res_theo = (2*(a**sp.Rational(1, 2))*(r**sp.Rational(-1))*k1)

    rho = sp.Rational(-1, 2)
    mu = sp.Rational(1, 2)
    res = IntegralAA((rho, mu, a, r), (x, a, np.inf)).doit()

    # Tests
    assert (res.equals(res_theo))


@pytest.mark.integral
def test_particular_integralAA_3():
    r"""Should fail if the obtained result does not correspond to
    theoretical expectation.

    Notes
    -----
    Represents the following integral:

    .. math::  \int_{a}^{\infty} d t\, e^{-r \sqrt{t}}\,
               t^{-\frac{3}{2}} \big(t - a\big)^{\frac{1}{2}}
               = \pi a^{\frac{1}{2}} r + 2 K_0(a^{\frac{1}{2}}r)
               -2ra^{\frac{1}{2}} K_1(a^{\frac{1}{2}} r)
               - \pi a r^2 \Big(K_0(a^{\frac{1}{2}} r)
               \boldsymbol{L}_{-1}(a^{\frac{1}{2}} r)+K_1(a^{\frac{1}{2}} r)
               \boldsymbol{L}_0(a^{\frac{1}{2}} r)\Big)

    """

    a, r, x = sp.symbols("a r dummyx")
    b_arg = r*(a**sp.Rational(1, 2))
    klkl = KLKLFunction(b_arg.expand())
    k0 = sp.besselk(0, b_arg)
    k1 = sp.besselk(1, b_arg)
    res_theo = ((2*k0) - (2*(a**sp.Rational(1, 2))*r*k1)
                - (sp.pi*a*(r**sp.Rational(2))*klkl)
                + (sp.pi*(a**sp.Rational(1, 2))*r)
               )

    rho = sp.Rational(-3, 2)
    mu = sp.Rational(1, 2)
    res = IntegralAA((rho, mu, a, r), (x, a, np.inf)).doit()

    # Tests
    assert (res.equals(res_theo))


@pytest.mark.integral
def test_particular_integralAA_4():
    r"""Should fail if the obtained result does not correspond to
    theoretical expectation.

    Notes
    -----
    Represents the following integral:

    .. math::  \int_{a}^{\infty} d t\, e^{-r \sqrt{t}}\, t^{-\frac{5}{2}}
               \big(t - a\big)^{\frac{1}{2}} = \frac{r^2\pi}{2}
               \bigg(1 - \frac{a r^2}{3}\bigg) \Big(K_0(a^{\frac{1}{2}} r)
               \boldsymbol{L}_{-1}(a^{\frac{1}{2}} r)+ K_1(a^{\frac{1}{2}} r)
               \boldsymbol{L}_0(a^{\frac{1}{2}} r)\Big)\\
               + \frac{1}{6} r\pi a^{-\frac{1}{2}} \big(a r^2 -3\big)
               + \frac{r^2}{3} K_0(a^{\frac{1}{2}}r)
               + \frac{1}{3} ra^{-\frac{1}{2}} \big(2-r^2 a\big)
               K_1(a^{\frac{1}{2}} r)

    """

    a, r, x = sp.symbols("a r dummyx")
    b_arg = r*(a**sp.Rational(1, 2))
    klkl = KLKLFunction(b_arg.expand())
    k0 = sp.besselk(0, b_arg)
    k1 = sp.besselk(1, b_arg)

    res_theo = (((r**sp.Rational(2))*sp.Rational(1, 2)*sp.pi
                 *(1-(a*sp.Rational(1, 3)*(r**sp.Rational(2))))*klkl)
                + (sp.Rational(1, 6)*r*sp.pi*(a**sp.Rational(-1,2))
                   *(a*(r**sp.Rational(2))-3))
                + (sp.Rational(1,3)*(r**sp.Rational(2))*k0)
                + (sp.Rational(1,3)*r*(a**sp.Rational(-1,2))
                   *(2-a*(r**sp.Rational(2)))*k1)
               )

    rho = sp.Rational(-5, 2)
    mu = sp.Rational(1, 2)
    res = IntegralAA((rho, mu, a, r), (x, a, np.inf)).doit()

    # Tests
    assert (res.equals(res_theo))


@pytest.mark.integral
@pytest.mark.parametrize("rho, mu",
    [(sp.Rational(1, 2), sp.Rational(1, 2)),
     (sp.Rational(-1, 2), sp.Rational(1, 2)),
     (sp.Rational(-3, 2), sp.Rational(1, 2)),
     (sp.Rational(-5, 2), sp.Rational(1, 2)),
     (sp.Rational(-5, 2), sp.Rational(1, 2)),
     (sp.Rational(0), sp.Rational(1, 2)),
     (sp.Rational(-1), sp.Rational(-1, 2)),
     (sp.Rational(-1), sp.Rational(-3, 2)),
     (sp.Rational(-3, 2), sp.Rational(-3, 2)),
    ])
def test_eval_integralAA_1(rho, mu):
    r"""Should fail if the obtained result does not correspond to
    numerical integration within a certain error margin.
    """
    a_num = 0.01
    r_num = 0.5
    a, r = sp.symbols("a r")
    res = IntegralAA((rho, mu, a, r), (r, a, np.inf)).doit()
    res = res.subs({a: a_num, r: r_num})
    res_num = res.evalf()

    def integrand(x, rho, mu, a, r):

        return FunctionAA(x, rho, mu, a, r).doit().evalf()

    res_int = quad(integrand, a_num, np.inf, args=(rho, mu, a_num, r_num))[0]

    # Tests
    assert (math.isclose(res_num, res_int, rel_tol=1e-4))
