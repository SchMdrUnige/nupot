
""".. moduleauthor:: Sacha Medaer"""

import numpy as np
import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.integrals.integralAA import IntegralAA
from nupot.potentials.nu_pair_potential import NuPairPotential


class DiracNuPairPotential(NuPairPotential):
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
               \Biggr[ 1-\frac{1}{t}\overline{m_{ij}^2}-\frac{1}{2}
               \biggr[\frac{\Delta m_{ij}^2}{t}\biggr]^2\Biggr]
    """
    # ==================================================================
    @staticmethod
    def pot_term(m_1, m_2, r, Q_A, Q_B, orders = [1]*3)-> cst.EXPR_TYPE:

        if (m_1.equals(m_2)):
            return (Q_A*Q_B*(cst.G_F**sp.Rational(2))*(m_1**sp.Rational(3))
                    *sp.simplify(sp.besselk(3, 2*m_1*r))
                    /(16*(sp.pi**sp.Rational(3))*(r**sp.Rational(2))))

        else:
            term_1 = DiracNuPairPotential.term_1(m_1, m_2, r, orders[0])
            term_2 = DiracNuPairPotential.term_2(m_1, m_2, r, orders[1])
            term_3 = DiracNuPairPotential.term_3(m_1, m_2, r, orders[2])

            return ((Q_A*Q_B*(cst.G_F**sp.Rational(2))/(sp.pi**sp.Rational(3)))
                    * (term_1 + term_2 + term_3))
    # ==================================================================
    @staticmethod
    def term_1(m_1, m_2, r, order: int = 1):
        expr = sp.Rational(0)
        a = (m_1 + m_2)**sp.Rational(2)
        e = (m_1 - m_2)**sp.Rational(2)
        dummyx = sp.symbols('dummyx')
        for k in range(order):
            integral = IntegralAA((sp.Rational(0.5-k), sp.Rational(1, 2),
                                   a, r), (dummyx, a, np.inf))
            expr += ((sp.Rational(-1)**k)
                     * sp.Rational(sp.binomial(sp.Rational(1, 2), k))
                     * (e**k) * sp.Rational(1, 192) * (1/r)
                     * integral.doit())

        return expr
    # ==================================================================
    @staticmethod
    def term_2(m_1, m_2, r, order: int = 1):
        expr = sp.Rational(0)
        a = (m_1 + m_2)**sp.Rational(2)
        e = (m_1 - m_2)**sp.Rational(2)
        f = (m_1**sp.Rational(2)) + (m_2**sp.Rational(2))
        dummyx = sp.symbols('dummyx')
        for k in range(order):
            integral = IntegralAA((sp.Rational(-0.5-k), sp.Rational(1, 2),
                                   a, r), (dummyx, a, np.inf))
            expr -= ((sp.Rational(-1)**k)
                     * sp.Rational(sp.binomial(sp.Rational(1, 2), k))
                     * (e**k) * f * sp.Rational(1, 384) * (1/r)
                     * integral.doit())

        return expr
    # ==================================================================
    @staticmethod
    def term_3(m_1, m_2, r, order: int = 1):
        expr = sp.Rational(0)
        a = (m_1 + m_2)**sp.Rational(2)
        e = (m_1 - m_2)**sp.Rational(2)
        g = (m_1**sp.Rational(2)) - (m_2**sp.Rational(2))
        dummyx = sp.symbols('dummyx')
        for k in range(order):
            integral = IntegralAA((sp.Rational(-1.5-k), sp.Rational(1, 2),
                                   a, r), (dummyx, a, np.inf))
            expr -= ((sp.Rational(-1)**k)
                     * sp.Rational(sp.binomial(sp.Rational(1, 2), k))
                     * (e**k) * (g**sp.Rational(2))
                     * sp.Rational(1, 384) * (1/r)
                     * integral.doit())

        return expr



if __name__ == '__main__':
    from sympy import symbols
    from nupot.physics.atom import Atom

    m_1, m_2, m_3, r, a, b = symbols('m_1 m_2 m_3 r a b', real=True, positive=True)
    atom_A = Atom('Fe')
    atom_B = Atom('Cu')
    nuV = DiracNuPairPotential(atom_A, atom_B, m_1, m_2, m_3, r)
    expr_nuV = nuV.doit()
    new_expr = expr_nuV.subs({m_1: 0.1, m_2: 0.1, m_3: 0.1, r: 0.5, a: 1.0})
    print('numeracil value : ', new_expr.evalf())
    #fact = DiracNuPairPotential.factorize(expr_nuV, m_1, m_2, m_3, r)
    over_r = DiracNuPairPotential.integrate_over_r(expr_nuV, m_1, m_2, m_3, r,
                                                   a, np.inf)

    new_expr = over_r.subs({m_1: 0.1, m_2: 0.1, m_3: 0.1, a: 0.001})
    print('numeracil value : ', new_expr.evalf())

    #for elem in fact:
    #    print('\n\n ---------------------------------')
    #    print(elem)
