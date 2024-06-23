
""".. moduleauthor:: Sacha Medaer"""

import numpy as np
import sympy as sp
from itertools import combinations_with_replacement

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.functions.klkl_function import KLKLFunction
from nupot.integrals.integralBA import IntegralBA
from nupot.integrals.integralBB import IntegralBB
from nupot.integrals.integralBC import IntegralBC
from nupot.integrals.integralBD import IntegralBD
from nupot.integrals.integralBE import IntegralBE
from nupot.integrals.integralBF import IntegralBF
from nupot.integrals.integralBG import IntegralBG
from nupot.matrices.global_weak_charge import GlobalWeakCharge
from nupot.physics.atom import Atom
from nupot.potentials.abstract_potential import AbstractPotential


class NuPairPotentialNotImplementedError(NotImplementedError):
    pass


class NuPairPotential(sp.Function):
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
               \Biggr[ 1-\frac{1}{t}M-\frac{1}{2}
               \biggr[\frac{\Delta m_{ij}^2}{t}\biggr]^2\Biggr]
               \text{with}\quad M = \begin{cases}
                \overline{m_{ij}^2}
                & \quad \text{if considering Dirac neutrinos}\\
                \overline{m_{ij}^2}  + 3m_im_j
                & \quad \text{if considering Majorana neutrinos}
              \end{cases}

    """
    def __new__(cls, atom_A: Atom, atom_B: Atom, *args, **kwargs):
        # pass along the parameters only to the parent class

        return super().__new__(cls, *args, **kwargs)
    # ==================================================================
    def __init__(self, atom_A: Atom, atom_B: Atom, *agrs, **kwargs):
        self._atom_A: Atom = atom_A
        self._atom_B: Atom = atom_B
    # ==================================================================
    @property
    def atom_A(self):

        return self._atom_A
    # ==================================================================
    @property
    def atom_B(self):

        return self._atom_B
    # ==================================================================
    def doit(self, deep=False, **hints):
        # Extracting arguments
        m_1, m_2, m_3, r = self.args
        # Recursively call doit with the deep parameter
        if (deep):
            m_1 = m_1.doit(deep=deep, **hints)
            m_2 = m_2.doit(deep=deep, **hints)
            m_3 = m_3.doit(deep=deep, **hints)
            r = r.doit(deep=deep, **hints)
        # Group masses in iterable
        masses = (m_1, m_2, m_3)
        # Initiate the global weak charge matrix
        Q_A = GlobalWeakCharge(self._atom_A.atomic_number,
                               self._atom_A.neutrons)
        Q_B = GlobalWeakCharge(self._atom_B.atomic_number,
                               self._atom_B.neutrons)
        # Calculate potential
        V = sp.Rational(0)
        for comb in list(combinations_with_replacement([0, 1, 2], 2)):
            crt_Q_A = Q_A[comb[0], comb[1]]
            crt_Q_B = sp.conjugate(Q_B[comb[0], comb[1]])
            V += self.pot_term(masses[comb[0]], masses[comb[1]], r,
                               crt_Q_A, crt_Q_B)

        return V
    # ==================================================================
    def _eval_evalf(self, prec) -> sp.Float:

        return self.doit().evalf()
    # ==================================================================
    @staticmethod
    def factorize(expr, m_1, m_2, m_3, r):
        #m_1, m_2, m_3, r = self.args
        vars = []
        vars.append(2*m_1*r)
        vars.append(2*m_2*r)
        vars.append(2*m_3*r)
        # Must expand for the factorization logic, assumed that the
        # argument has been expanded in custom function
        vars.append((r*(m_1 + m_2)).expand())
        vars.append((r*(m_2 + m_3)).expand())
        vars.append((r*(m_1 + m_3)).expand())
        factors = []
        for i, var in enumerate(vars):
            factors.append(sp.besselk(0, var))
            factors.append(sp.besselk(1, var))
            if (i > 2): # not KLKLFunction(2*m_i*r)
                factors.append(KLKLFunction(var))

        factorization = []
        expr_ = expr.expand()
        for factor in factors:
            coeff_expr = expr_.coeff(factor)
            expr_ = expr_.subs(factor, 0)
            if (coeff_expr):
                factorization.append((coeff_expr, factor))
        if (expr_):
            factorization.append((expr_, sp.Rational(1))) # add the rest

        return factorization
    # ==================================================================
    @staticmethod
    def integrate_over_r(expr, m_1, m_2, m_3, r, a, b):
        fact_expr = NuPairPotential.factorize(expr, m_1, m_2, m_3, r)
        res = sp.Rational(0)
        for fact in fact_expr:
            if (isinstance(fact[1], sp.besselk)):
                if (fact[1].order == 0):
                    alpha = fact[1].argument.coeff(r)
                    for arg in fact[0].args:
                        coeff, expo = util.get_coeff_and_expo_of_symbol(arg, r)
                        if (expo % 2):  # odd
                            if (expo > 0):
                                n = (expo - 1) // 2
                                integral = IntegralBB((n, alpha), (r, a, b))
                            else:
                                n = (abs(expo) - 1) // 2
                                integral = IntegralBE((n, alpha), (r, a, b))
                            res += coeff * integral.doit()
                        else:   # even
                            if (expo > 0):
                                n = expo // 2
                                integral = IntegralBF((n, alpha), (r, a, b))
                            else:

                                raise NuPairPotentialNotImplementedError()
                            res += coeff * integral.doit()
                elif (fact[1].order == 1):
                    alpha = fact[1].argument.coeff(r)
                    for arg in fact[0].args:
                        coeff, expo = util.get_coeff_and_expo_of_symbol(arg, r)
                        if (not (expo % 2)):  # even
                            if (expo > 0):
                                n = expo // 2
                                integral = IntegralBC((n, alpha), (r, a, b))
                            else:
                                n = abs(expo) // 2
                                integral = IntegralBD((n, alpha), (r, a, b))
                            res += coeff * integral.doit()
                        else:
                            if (expo > 0):
                                n = (expo - 1) // 2
                                integral = IntegralBG((n, alpha), (r, a, b))
                            else:

                                raise NuPairPotentialNotImplementedError()
                            res += coeff * integral.doit()
                else:

                    raise NuPairPotentialNotImplementedError()
            elif (isinstance(fact[1], KLKLFunction)):
                alpha = fact[1].argument.coeff(r)
                for arg in fact[0].args:
                    coeff, expo = util.get_coeff_and_expo_of_symbol(arg, r)
                    if (expo % 2):  # odd
                        if (expo > 0):
                            n = (expo - 1) // 2
                            integral = IntegralBA((n, alpha), (r, a, b))
                            res += coeff * integral.doit()
                        else:

                            raise NuPairPotentialNotImplementedError()
                    else:

                        NuPairPotentialNotImplementedError()

            else:   # consider all other symbol/functions as constant
                #res += sp.Rational(0)#sp.integrate(fact[0], r)
                print('')

        return res
