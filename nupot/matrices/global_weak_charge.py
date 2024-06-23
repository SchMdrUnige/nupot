
""".. moduleauthor:: Sacha Medaer"""

import sympy as sp

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.physics.nu import Nu


class GlobalWeakCharge(sp.Matrix):
    """This class represents the global weak charge of the neutrino
    pair mediated matter scattering.
    """

    def __new__(cls, *args, **kwargs):
        Z, N = args
        pmns = Nu.PMNSMatrix
        elements = [[] for _ in range(3)]
        elements[0].append(2*Z*pmns[0, 0]*sp.conjugate(pmns[0, 0]) - N)
        elements[0].append(2*Z*pmns[0, 0]*sp.conjugate(pmns[0, 1]))
        elements[0].append(2*Z*pmns[0, 0]*sp.conjugate(pmns[0, 2]))
        elements[1].append(2*Z*pmns[0, 1]*sp.conjugate(pmns[0, 0]))
        elements[1].append(2*Z*pmns[0, 1]*sp.conjugate(pmns[0, 1]) - N)
        elements[1].append(2*Z*pmns[0, 1]*sp.conjugate(pmns[0, 2]))
        elements[2].append(2*Z*pmns[0, 2]*sp.conjugate(pmns[0, 0]))
        elements[2].append(2*Z*pmns[0, 2]*sp.conjugate(pmns[0, 1]))
        elements[2].append(2*Z*pmns[0, 2]*sp.conjugate(pmns[0, 2]) - N)

        return super().__new__(cls, elements, **kwargs)



if __name__ == '__main__':
    import sympy as sp
    from nupot.physics.atom import Atom

    atom_fe = Atom('Fe')
    atom_cu = Atom('Cu')
    Q_A = GlobalWeakCharge(atom_fe.atomic_number, atom_fe.neutrons)
    Q_B = GlobalWeakCharge(atom_cu.atomic_number, atom_cu.neutrons)
    Q_01 = Q_A[0, 1]*sp.conjugate(Q_B[0, 1])
    print('elem : ', Q_A[0, 1])
    print(Q_01)
    print(Q_01.evalf())

    Q_11 = Q_A[1, 1]*sp.conjugate(Q_B[1, 1])
    print('elem : ', Q_A[1, 1])
    print(Q_11)
    print(Q_11.evalf())
