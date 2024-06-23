
""".. moduleauthor:: Sacha Medaer"""

import numpy as np
import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.integrals.integralAA import IntegralAA
from nupot.potentials.nu_pair_potential import NuPairPotential
from nupot.potentials.dirac_nu_pair_potential import DiracNuPairPotential


class MajoranaNuPairPotential(NuPairPotential):
    r"""

    Notes
    -----
    Represents the following potential arising from the long force of
    the neutrino-pair matter scattering:

    .. math::  V(r) = \frac{-1}{4\pi^2r}\sum_{ij}\int_{t_{ij}}^{\infty}
                      d t\, e^{-r \sqrt{t}}
                      \operatorname{Im}{\mathcal{M}_{ij}(t)}
               \text{with}\quad \operatorname{Im}{\mathcal{M}_{ij}}
               = -\frac{G^2_F}{48\pi}\, t\, Q^{ij}_{W,A}Q^{ij\, *}_{W,B}
               \sqrt{1-\frac{4\overline{m^2_{ij}}}{t}
               +\biggr[\frac{\Delta m^2_{ij}}{t}\biggr]^2}\, \times\,
               \Biggr[ 1-\frac{1}{t}(\overline{m_{ij}^2}  + 3m_im_j)
               -\frac{1}{2}
               \biggr[\frac{\Delta m_{ij}^2}{t}\biggr]^2\Biggr]

    """
    # ==================================================================
    @staticmethod
    def pot_term(m_1, m_2, r, Q_A, Q_B, orders = [3]*4)-> cst.EXPR_TYPE:

        if (m_1.equals(m_2)):
            return (Q_A*Q_B*(cst.G_F**sp.Rational(2))*(m_1**sp.Rational(2))
                    *sp.simplify(sp.besselk(2, 2*m_1*r))
                    /(8*(sp.pi**sp.Rational(3))*(r**sp.Rational(3))))

        else:
            term_1 = MajoranaNuPairPotential.term_1(m_1, m_2, r, orders[2])
            term_1 *= Q_A*Q_B*(cst.G_F**sp.Rational(2))/(sp.pi**sp.Rational(3))
            term_2 = DiracNuPairPotential.pot_term(m_1, m_2, r, Q_A, Q_B,
                                                   orders[:2] + orders[3:])

            return term_1 + term_2
    # ==================================================================
    @staticmethod
    def term_1(m_1, m_2, r, order: int = 1):
        expr = sp.Rational(0)
        a = (m_1 + m_2)**sp.Rational(2)
        e = (m_1 - m_2)**sp.Rational(2)
        h = m_1 * m_2
        dummyx = sp.symbols('dummyx')
        for k in range(order):
            integral = IntegralAA((sp.Rational(-0.5-k), sp.Rational(1, 2),
                                   a, r), (dummyx, a, np.inf))
            expr -= ((sp.Rational(-1)**k)
                     * sp.Rational(sp.binomial(sp.Rational(1, 2), k))
                     * (e**k) * h * sp.Rational(1, 64) * (1/r)
                     * integral.doit())

        return expr
