import pytest

import numpy as np
import sympy as sp

from nupot.functions.klkl_function import KLKLFunction
import nupot.utils.constants as cst
import nupot.utils.utilities as util

# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------

@pytest.mark.functions
def test_particular_values():
    r"""Should fail if the eval does not match the theoretical
    expectation for some particular values of the argument.
    """
    # Tests
    assert (KLKLFunction(0) == np.inf)
    assert (KLKLFunction(np.inf) == 0)


@pytest.mark.functions
def test_equality():
    r"""Should fail if two supposedly equal functions are not assert
    as equal.
    """
    a, b, c = sp.symbols('a b c')
    klkl1 = KLKLFunction(a*(b+c))
    klkl2 = KLKLFunction(a*(b+c))
    klkl3 = KLKLFunction(a*b + a*c)
    # Tests
    assert (klkl1.equals(klkl2))
    assert (klkl1.equals(klkl3))


@pytest.mark.functions
def test_factorization():
    r"""Should fail if the function can not be factorize in an
    expression.
    """
    a, b, c, d, e = sp.symbols('a b c d e')
    klkl1 = KLKLFunction(a*(b+c))
    klkl2 = KLKLFunction(a*(b+c))
    expr = ((d**2) * klkl1) + (e * klkl2)
    coeff_expr = expr.coeff(KLKLFunction(a*(b+c)))
    expected_coeff = (d**2) + e
    # Tests
    assert (coeff_expr.equals(expected_coeff))
