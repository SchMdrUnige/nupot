
""".. moduleauthor:: Sacha Medaer"""

import copy

import sympy as sp


def get_coeff_and_expo_of_symbol(expr, r):
    """Return the ceofficient and the exponential of the symbol r in
    the expression expr."""
    if (expr.has(r)):
        r_coeff = expr.subs(r, 1)
        r_var = expr.subs(r_coeff, 1)
        if (isinstance(r_var, sp.core.power.Pow)):
            r_exp = r_var.args[1]
        else:   # isinstance(r, sp.core.symbol.Symbol)
            r_exp = 1
    else:
        r_coeff = copy.copy(expr)
        r_exp = 0

    return r_coeff, r_exp
