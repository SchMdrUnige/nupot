import pytest

import math
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.special import kn

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.integrals.ksi_integral import KSiFunction, KSiIntegral

# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------


@pytest.mark.function
@pytest.mark.parametrize("r_2_num",
    [1e0, np.inf]
    )
def test_eval_integralBA_1(r_2_num):
    r"""Should fail if the obtained result does not correspond to
    numerical integration within a certain error margin.
    """
    a_num = 0.5
    r_1_num = 1e-4
    a, r = sp.symbols("a r")
    res = KSiIntegral((a, ), (r, r_1_num, r_2_num)).doit()
    res = res.subs({a: a_num})
    res_num = res.evalf()

    def integrand(r, a):

        return KSiFunction(r, a).doit().evalf()

    res_int = quad(integrand, r_1_num, r_2_num, args=(a_num, ))[0]

    # Tests
    assert (math.isclose(res_num, res_int, rel_tol=1e-3))
