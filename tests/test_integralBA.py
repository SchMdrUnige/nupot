import pytest

import math
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.special import kn

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.integrals.integralBA import FunctionBA, IntegralBA
from nupot.functions.klkl_function import KLKLFunction

# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------


@pytest.mark.integral
@pytest.mark.parametrize("n_num, r_2_num",
    [(1, 1.0), (2, 1e0), (3, 1e0), (4, 1e0), (5, 1e0),
    ])
def test_eval_integralBA_1(n_num, r_2_num):
    r"""Should fail if the obtained result does not correspond to
    numerical integration within a certain error margin.
    """
    a_num = 1.0
    r_1_num = 1e-4
    a, r = sp.symbols("a r")
    n = sp.Rational(n_num)
    res = IntegralBA((n, a), (r, r_1_num, r_2_num)).doit()
    res = res.subs({a: a_num})
    res_num = res.evalf()

    def integrand(r, a, n):

        return FunctionBA(r, n, a).doit().evalf()

    res_int = quad(integrand, r_1_num, r_2_num, args=(a_num, n_num))[0]

    # Tests
    assert (math.isclose(res_num, res_int, rel_tol=1e-4))
