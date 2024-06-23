
""".. moduleauthor:: Sacha Medaer"""

import math

import numpy as np
import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.symbols.constant_real_symbol import ConstantRealSymbol


def build_pmns_matrix(theta_12, theta_23, theta_13, delta_cp):
    # Initiate the parameters of the PMNS matrix which rely on the
    # values of the CP Violation phase and the mixing angles
    c_12 = sp.cos(theta_12)
    c_23 = sp.cos(theta_23)
    c_13 = sp.cos(theta_13)
    s_12 = sp.sin(theta_12)
    s_23 = sp.sin(theta_23)
    s_13 = sp.sin(theta_13)
    # Define matrix elements
    elements = [[] for _ in range(3)]
    elements[0].append(c_12*c_13)
    elements[0].append(s_12*c_13)
    elements[0].append(s_13*sp.exp(-sp.I*delta_cp))
    elements[1].append((-s_12*c_23)
                       - (c_12*s_13*s_23*sp.exp(sp.I*delta_cp)))
    elements[1].append((c_12*c_23)
                       - (s_12*s_13*s_23*sp.exp(sp.I*delta_cp)))
    elements[1].append(c_13*s_23)
    elements[2].append((s_12*s_23)
                       - (c_12*s_13*c_23*sp.exp(sp.I*delta_cp)))
    elements[2].append((c_12*s_23)
                       - (s_12*s_13*c_23*sp.exp(sp.I*delta_cp)))
    elements[2].append(c_13*c_23)

    return sp.Matrix(elements)


class NuInputError(Exception):
    pass


class classproperty(property):

    def __get__(self, owner_self, owner_cls):

        return self.fget(owner_cls)


class Nu(object):
    """This class contains the neutrino properties.
    """
    # from nufit.org (value, error_low, error_high)
    # Mixing angles :
    __theta_12 = (ConstantRealSymbol(math.radians(33.41), r'\theta_{12}',
                                     positive=True),
                  ConstantRealSymbol(math.radians(0.72), 'theta_12_unc_l',
                                     positive=True),
                  ConstantRealSymbol(math.radians(0.75), 'theta_12_unc_u',
                                     positive=True))
    __theta_23 = (ConstantRealSymbol(math.radians(49.1), r'\theta_{23}',
                                     positive=True),
                  ConstantRealSymbol(math.radians(1.3), 'theta_23_unc_l',
                                     positive=True),
                  ConstantRealSymbol(math.radians(1.0), 'theta_23_unc_u',
                                     positive=True))
    __theta_13 = (ConstantRealSymbol(math.radians(8.54), r'\theta_{13}',
                                     positive=True),
                  ConstantRealSymbol(math.radians(0.12), 'theta_13_unc_l',
                                     positive=True),
                  ConstantRealSymbol(math.radians(0.11), 'theta_13_unc_u',
                                     positive=True))
    # CP Violation Phase :
    __delta_cp = (ConstantRealSymbol(math.radians(197), r'\delta_{CP}',
                                     positive=True),
                  ConstantRealSymbol(math.radians(25), 'delta_cp_unc_l',
                                     positive=True),
                  ConstantRealSymbol(math.radians(42), 'delta_cp_unc_u',
                                     positive=True))
    # Create PMNS Matrix
    __pmns_matrix = build_pmns_matrix(__theta_12[0], __theta_23[0],
                                      __theta_13[0], __delta_cp[0])
    __pmns_matrix_lower_unc = build_pmns_matrix(__theta_12[0] - __theta_12[1],
                                                __theta_23[0] - __theta_23[1],
                                                __theta_13[0] - __theta_13[1],
                                                __delta_cp[0] - __delta_cp[1])
    __pmns_matrix_upper_unc = build_pmns_matrix(__theta_12[0] + __theta_12[2],
                                                __theta_23[0] + __theta_23[2],
                                                __theta_13[0] + __theta_13[2],
                                                __delta_cp[0] + __delta_cp[2])
    # |Delta m_{21}^2 : signed value # in eV^2
    __delta_m_sq_21 = (ConstantRealSymbol(7.49e-5, r'\Delta m_{21}^2',
                                          positive=True),
                       ConstantRealSymbol(0.17e-5, 'delta_m_sq_21_unc_l',
                                          positive=True),
                       ConstantRealSymbol(0.19e-5, 'delta_m_sq_21_unc_u',
                                          positive=True))
    # |\Delta m_{31}^2| : unsigned value # in eV^2
    __delta_m_sq_31 = (ConstantRealSymbol(2.484e-3, r'\Delta m_{31}^2'),
                       ConstantRealSymbol(0.048e-3, 'delta_m_sq_31_unc_l'),
                       ConstantRealSymbol(0.045e-3, 'delta_m_sq_31_unc_u'))

    @classproperty
    def theta_12(cls) -> ConstantRealSymbol:

        return cls.__theta_12[0]

    @classproperty
    def theta_12_unc(cls) -> tuple[ConstantRealSymbol, ConstantRealSymbol]:

        return (cls.__theta_12[1], cls.__theta_12[2])

    @classproperty
    def theta_23(cls) -> ConstantRealSymbol:

        return cls.__theta_23[0]

    @classproperty
    def theta_23_unc(cls) -> tuple[ConstantRealSymbol, ConstantRealSymbol]:

        return (cls.__theta_23[1], cls.__theta_23[2])

    @classproperty
    def theta_13(cls) -> ConstantRealSymbol:

        return cls.__theta_13[0]

    @classproperty
    def theta_13_unc(cls) -> tuple[ConstantRealSymbol, ConstantRealSymbol]:

        return (cls.__theta_13[1], cls.__theta_13[2])

    @classproperty
    def delta_cp(cls) -> ConstantRealSymbol:

        return cls.__delta_cp[0]

    @classproperty
    def delta_cp_unc(cls) -> tuple[ConstantRealSymbol, ConstantRealSymbol]:

        return (cls.__delta_cp[1], cls.__delta_cp[2])

    @classproperty
    def PMNSMatrix(cls) -> sp.Matrix:

        return cls.__pmns_matrix

    @classproperty
    def PMNSMatrix_lower_unc(cls) -> sp.Matrix:

        return cls.__pmns_matrix_lower_unc

    @classproperty
    def PMNSMatrix_upper_unc(cls) -> sp.Matrix:

        return cls.__pmns_matrix_upper_unc

    @classproperty
    def delta_m_sq_21(cls) -> ConstantRealSymbol:

        return cls.__delta_m_sq_21[0]

    @classproperty
    def delta_m_sq_21_unc(cls
                          ) -> tuple[ConstantRealSymbol, ConstantRealSymbol]:

        return (cls.__delta_m_sq_21[1], cls.__delta_m_sq_21[2])

    @classproperty
    def delta_m_sq_31(cls) -> ConstantRealSymbol:

        return cls.__delta_m_sq_31[0]

    @classproperty
    def delta_m_sq_31_unc(cls
                          ) -> tuple[ConstantRealSymbol, ConstantRealSymbol]:

        return (cls.__delta_m_sq_31[1], cls.__delta_m_sq_31[2])

    @staticmethod
    def get_masses_from_smallest_mass(m_0: float, hierarchy = 'NH'
                                      ) -> tuple[ConstantRealSymbol]:
        # max value allowed for m_0

        if (hierarchy == 'NH'): # NH: m_1 < m_2 << m_3 -> delta_m_sq_31 > 0
            m_1 = ConstantRealSymbol(m_0, 'm_1', positive=True)
            m_2_value = np.sqrt(float(Nu.delta_m_sq_21.evalf()) + (m_0**2))
            m_2 = ConstantRealSymbol(m_2_value, 'm_2', positive=True)
            m_3_value = np.sqrt(float(Nu.delta_m_sq_31.evalf()) + (m_0**2))
            m_3 = ConstantRealSymbol(m_3_value, 'm_3', positive=True)
        elif (hierarchy == 'IH'):   # IH: m_3 << m_1 < m_2 -> delta_m_sq_31 < 0
            m_3 = ConstantRealSymbol(m_0, 'm_3', positive=True)
            m_1_value = np.sqrt(float(Nu.delta_m_sq_31.evalf()) + (m_0**2))
            m_1 = ConstantRealSymbol(m_1_value, 'm_1', positive=True)
            m_2_value = np.sqrt(float(Nu.delta_m_sq_21.evalf()) + (m_1_value**2))
            m_2 = ConstantRealSymbol(m_2_value, 'm_2', positive=True)
        else:

            raise NotImplementedError()

        return (m_1, m_2, m_3)


if __name__ == '__main__':

    delta_cp_1 = Nu.delta_cp
    delta_cp_2 = Nu.delta_cp
    print(delta_cp_1 == delta_cp_2)

    m_0 = 0. # eV
    masses = Nu.get_masses_from_smallest_mass(m_0, hierarchy='NH')
    print('with NH: ')
    for i, mass in enumerate(masses):
        print(i, ' : ', mass.evalf(), ' eV')
    masses = Nu.get_masses_from_smallest_mass(m_0, hierarchy='IH')
    print('with IH: ')
    for i, mass in enumerate(masses):
        print(i, ' : ', mass.evalf(), ' eV')
