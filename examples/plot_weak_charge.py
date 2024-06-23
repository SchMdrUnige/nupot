
""".. moduleauthor:: Sacha Medaer"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from nupot.physics.atom import Atom
from nupot.matrices.global_weak_charge import GlobalWeakCharge

i = [0, 1, 2, 0, 0, 1]
j = [0, 1, 2, 1, 2, 2]
max_Z = 50
gwc_obj = []
for k in range(1, max_Z+1):
    atom = Atom(k)
    gwc_obj.append(GlobalWeakCharge(atom.atomic_number, atom.neutrons))


gwc_values = [np.zeros((max_Z, max_Z)) for _ in range(len(i))]
for m in range(len(i)):
    for k in range(1, max_Z+1):
        for l in range(1, max_Z+1):
            gwc_1 = gwc_obj[k-1]
            gwc_2 = gwc_obj[l-1]
            gwc_values[m][k-1, l-1] = (gwc_1[i[m],j[m]]
                                       * sp.conjugate(gwc_2[i[m],j[m]])
                                      ).evalf()

print(gwc_values)

nbr_vertical = 3
nbr_horizontal = 2
fig, axs = plt.subplots(nbr_vertical, nbr_horizontal)
label_size = 30

for n in range(nbr_vertical):
    for p in range(nbr_horizontal):
        ax = axs[n][p]
        c = ax.pcolor(np.arange(max_Z)+1, np.arange(max_Z)+1,
                      gwc_values[(n*nbr_horizontal)+p],
                      cmap='PuBu_r')
        fig.colorbar(c, ax=ax)

        ax.set_ylabel(r'Z', size=label_size)
        ax.set_xlabel(r'Z', size=label_size)
        ax.tick_params(axis='both', labelsize=label_size-5)
        ax.text(max_Z//2, max_Z//2,
                r'(i,j) = ({}, {})'.format(i[(n*nbr_horizontal)+p]+1,
                                           j[(n*nbr_horizontal)+p]+1),
                size=label_size)
        #plt.legend(fontsize=label_size-5)

#plt.title("Global Weak Charges as a function of atom numbers",
#           size=label_size)
#fig.tight_layout()
plt.show()
