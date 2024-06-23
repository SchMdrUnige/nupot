
""".. moduleauthor:: Sacha Medaer"""

from typing import Union

from mendeleev import element

import nupot.utils.constants as cst
import nupot.utils.utilities as util
from nupot.symbols.constant_real_symbol import ConstantRealSymbol


# Exceptions
class AtomInputError(Exception):
    pass


class Atom(object):
    """This class contains the atom properties. Currently, it acts
    as a custom API of the mendeleev python library.
    """

    def __init__(self, symbol: Union[int, str]) -> None:
        self._symbol: Union[int, str] = symbol
        if (Atom.does_element_exist(symbol)):
            self._elem = element(symbol)
        else:

            raise AtomInputError('The specified symbol {} was not found.'
                                 .format(symbol))

        self._atomic_number: ConstantRealSymbol
        name_ = 'Z_{' + str(self.symbol)  + '}'
        self._atomic_number = ConstantRealSymbol(self._elem.atomic_number,
                                                 name_, positive=True)
        name_ = 'N_{' + str(self.symbol)  + '}'
        self._neutrons: ConstantRealSymbol
        self._neutrons = ConstantRealSymbol(self._elem.neutrons, name_,
                                            positive=True)


        return None

    @staticmethod
    def does_element_exist(symbol: str):
        try:
            found = element(symbol)

            return True
        except:

            return False

    @property
    def name(self) -> str:

        return self._elem.name

    @property
    def symbol(self) -> Union[int, str]:

        return self._elem.symbol

    @property
    def atomic_number(self) -> ConstantRealSymbol:

        return self._atomic_number

    @property
    def neutrons(self) -> ConstantRealSymbol:

        return self._neutrons


if __name__ == '__main__':
    import sympy as sp

    atom = Atom('Fe')
    print('The atomic number is: ', atom.atomic_number.evalf())
    print('The neutron number is: ', atom.neutrons.evalf())
