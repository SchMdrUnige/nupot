import pytest

import math
import numpy as np
import sympy as sp
from scipy.integrate import quad, dblquad
from scipy.special import kn

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.physics.atom import Atom
from nupot.potentials.dirac_nu_pair_potential import DiracNuPairPotential

# ----------------------------------------------------------------------
# Tests ----------------------------------------------------------------
# ----------------------------------------------------------------------


@pytest.mark.integral
@pytest.mark.parametrize("m_1_num, m_2_num, r_num",
    [(0.1, 0.1, 0.5), (0.1, 0.01, 0.5), (0.1, 0.1, 1.0), (0.1, 0.01, 1.0),
     (0.1, 0.1, 1.5), (0.1, 0.01, 1.5),
    ])
def test_eval_potential_1(m_1_num, m_2_num, r_num):
    r"""Should fail if the obtained result does not correspond to
    numerical integration within a certain error margin.
    """
    Q_A = 1.0
    Q_B = 1.0
    m_1, m_2, r = sp.symbols('m_1 m_2 r')
    res = DiracNuPairPotential.pot_term(m_1, m_2, r, Q_A, Q_B, [3, 3, 3])
    res_num = res.subs({m_1: m_1_num, m_2: m_2_num, r: r_num})
    res_num = res_num.evalf()

    def integrand(t, r, m_1, m_2):

        m_2_bar = (m_1**2 + m_2**2)/2.0
        m_2_delta = m_1**2 - m_2**2

        term_1 = 1 - (4*m_2_bar/t) + (m_2_delta/t)**2
        term_2 = 1 - (m_2_bar/t) - 0.5*(m_2_delta/t)**2

        return t * np.sqrt(term_1) * term_2 * np.exp(-1*r*(t**0.5))

    a = (m_1_num + m_2_num)**2
    output_int = quad(integrand, a, np.inf, args=(r_num, m_1_num, m_2_num))
    res_int = ((cst.G_F.evalf())**2)/(192*(np.pi**3)*r_num) * output_int[0]

    # Tests
    assert (math.isclose(res_num, res_int, rel_tol=1e-6))


@pytest.mark.integral
@pytest.mark.parametrize("m_1_num, m_2_num, r_1_num",
    [(0.1, 0.5, 0.01),# (0.1, 0.1, 0.01), (0.1, 0.01, 0.01), (0.1, 0.1, 0.1), (0.1, 0.01, 0.1),
     #(0.1, 0.1, 0.5), (0.1, 0.01, 0.5),
    ])
def test_eval_potential_2(m_1_num, m_2_num, r_1_num):
    r"""Should fail if the obtained result does not correspond to
    numerical integration within a certain error margin.
    """
    Q_A = 1.0
    Q_B = 1.0
    r_2_num = 1e3
    m_1, m_2, m_3, r, a = sp.symbols('m_1 m_2 m_3 r a', real=True,
                                     positive=True)
    res = DiracNuPairPotential.pot_term(m_1, m_2, r, Q_A, Q_B, [2]*3)
    res_over_r = DiracNuPairPotential.integrate_over_r(res, m_1, m_2, m_3, r,
                                                       a, r_2_num)
    res_num = res_over_r.subs({m_1: m_1_num, m_2: m_2_num, a: r_1_num})
    res_num = res_num.evalf()

    def integrand(t, r, m_1, m_2):

        m_2_bar = (m_1**2 + m_2**2)/2.0
        m_2_delta = m_1**2 - m_2**2

        term_1 = 1 - (4*m_2_bar/t) + (m_2_delta/t)**2
        term_2 = 1 - (m_2_bar/t) - 0.5*(m_2_delta/t)**2

        factor = ((cst.G_F.evalf())**2)/(192*(np.pi**3)*r)

        return factor * t * np.sqrt(term_1) * term_2 * np.exp(-1*r*(t**0.5))

    a = (m_1_num + m_2_num)**2
    output_int = dblquad(integrand, r_1_num, r_2_num, a, np.inf,
                         args=(m_1_num, m_2_num))
    res_int = output_int[0]

    print('-------------------------------------------' , res_num, res_int)

    # Tests
    assert (math.isclose(res_num, res_int, rel_tol=1e-0))
