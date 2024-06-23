import pytest

import math
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.special import kn

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.integrals.integralBC import FunctionBC, IntegralBC

# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------


@pytest.mark.integral
def test_particular_integralBC_1():
    r"""Should fail if the obtained result does not correspond to
    theoretical expectation.

    Notes
    -----
    Represents the following integral:

    .. math::  \int K_1(\alpha r) d r = -\frac{1}{\alpha}r K_0(\alpha r)

    """

    a, r = sp.symbols("a r")
    b_arg = r*a
    k0 = sp.besselk(0, b_arg)

    res_theo = -k0 / a

    n = sp.Rational(0)
    res = IntegralBC((n, a), (r,)).doit()

    # Tests
    assert (res.equals(res_theo))


@pytest.mark.integral
@pytest.mark.parametrize("n_num, r_2_num",
    [(1, 1e0), (2, 1e0), (3, 1e0), (4, 1e0), (5, 1e0),
     (1, np.inf), (2, np.inf), (3, np.inf), (4, np.inf), (5, np.inf)
    ])
def test_eval_integralBC_1(n_num, r_2_num):
    r"""Should fail if the obtained result does not correspond to
    numerical integration within a certain error margin.
    """
    a_num = 0.5
    r_1_num = 1e-4
    a, r = sp.symbols("a r")
    n = sp.Rational(n_num)
    res = IntegralBC((n, a), (r, r_1_num, r_2_num)).doit()
    res = res.subs({a: a_num})
    res_num = res.evalf()

    def integrand(r, a, n):

        return FunctionBC(r, n, a).doit().evalf()

    res_int = quad(integrand, r_1_num, r_2_num, args=(a_num, n_num))[0]

    # Tests
    assert (math.isclose(res_num, res_int, rel_tol=1e-4))
