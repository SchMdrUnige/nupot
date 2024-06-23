import pytest

import math
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.special import kn

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.physics.atom import Atom
from nupot.potentials.majorana_nu_pair_potential import MajoranaNuPairPotential

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
    res = MajoranaNuPairPotential.pot_term(m_1, m_2, r, Q_A, Q_B, [3, 3, 3, 3])
    res_num = res.subs({m_1: m_1_num, m_2: m_2_num, r: r_num})
    res_num = res_num.evalf()

    def integrand(t, r, m_1, m_2):

        m_2_bar = (m_1**2 + m_2**2)/2.0
        m_2_delta = m_1**2 - m_2**2

        term_1 = 1 - (4*m_2_bar/t) + (m_2_delta/t)**2
        term_2 = 1 - (m_2_bar/t) - (3*m_1*m_2/t) - 0.5*(m_2_delta/t)**2

        return t * np.sqrt(term_1) * term_2 * np.exp(-1*r*(t**0.5))

    a = (m_1_num + m_2_num)**2
    output_int = quad(integrand, a, np.inf, args=(r_num, m_1_num, m_2_num))
    res_int = ((cst.G_F.evalf())**2)/(192*(np.pi**3)*r_num) * output_int[0]

    # Tests
    assert (math.isclose(res_num, res_int, rel_tol=1e-6))
