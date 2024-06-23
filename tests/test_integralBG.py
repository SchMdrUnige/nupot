import pytest

import math
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.special import kn

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.integrals.integralBG import FunctionBG, IntegralBG

# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------



@pytest.mark.integral
@pytest.mark.parametrize("n_num, r_2_num",
    [(1, 1e0), (2, 1e0), (3, 1e0), (4, 1e0), (5, 1e0),
     #(1, np.inf),# (2, np.inf), (3, np.inf), (4, np.inf), (5, np.inf)
    ])
def test_eval_integralBG_1(n_num, r_2_num):
    r"""Should fail if the obtained result does not correspond to
    numerical integration within a certain error margin.
    """
    a_num = 0.5
    r_1_num = 1e-4
    a, r = sp.symbols("a r")
    n = sp.Rational(n_num)
    res = IntegralBG((n, a), (r, r_1_num, r_2_num)).doit()
    res = res.subs({a: a_num})
    res_num = res.evalf()

    def integrand(r, a, n):

        return FunctionBG(r, n, a).doit().evalf()

    res_int = quad(integrand, r_1_num, r_2_num, args=(a_num, n_num))[0]

    print('------------------------------ ', res_num, res_int)

    # Tests
    assert (math.isclose(res_num, res_int, rel_tol=1e-4))
