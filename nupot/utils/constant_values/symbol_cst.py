
""".. moduleauthor:: Sacha Medaer"""

from scipy.constants import physical_constants as pyc

from nupot.symbols.constant_real_symbol import ConstantRealSymbol

# Fermi Coupling Constant
G_F = ConstantRealSymbol(pyc['Fermi coupling constant'][0], 'G_F',
                         positive=True)
