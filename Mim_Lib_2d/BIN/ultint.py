# -*- coding: utf-8 -*-
"""
INTRO

@author: Yi Zhang （张仪）, Created on Thu Oct 26 17:19:31 2017
    Aerodynamics
    Faculty of Aerospace Engineering
    TU Delft
"""
import types
import numpy as np
from functionals import lobatto_quad, extended_gauss_quad, lagrange_basis, edge_basis, gauss_quad
# %% Mimetic basis function values
def mbfv(x, p, poly_type):
    assert np.min(x) >= -1 and np.max(x) <=1, "x should be in [-1,1]"
    assert p >= 1 and isinstance(p,int), "p should be positve integer"

    if poly_type == "LobN": # lobatto polynomials
        nodes, weights = lobatto_quad(p)
        basis=lagrange_basis(nodes, x)
        return  basis
    elif poly_type == "LobE": # lobatto edges functions
        nodes, weights = lobatto_quad(p)
        basis=edge_basis(nodes, x)
        return basis

    elif poly_type == "GauN":  # gauss polynomials
        nodes, weights = gauss_quad(p)
        basis=lagrange_basis(nodes, x)
        return basis
    elif poly_type == "GauE":  # gauss edges functions
        nodes, weights = gauss_quad(p)
        basis=edge_basis(nodes, x)
        return basis

    elif poly_type == "etGN": # extended-gauss polynomials
        nodes, weights = extended_gauss_quad(p)
        basis=lagrange_basis(nodes, x)
        return basis
    elif poly_type == "etGE": # extended-gauss edges functions
        nodes, weights = extended_gauss_quad(p)
        basis=edge_basis(nodes, x)
        return basis
    
    else:
        raise Exception("Error, poly_type wrong......")
        
# %%
def _size_check(basis):
    poly_type, p = basis
    if poly_type == 'lobatto_node': return p+1
    elif poly_type == 'lobatto_edge': return p
    elif poly_type == 'gauss_node': return p
    elif poly_type == 'gauss_edge': return p-1
    elif poly_type == 'ext_gauss_node': return p+2
    elif poly_type == 'ext_gauss_edge': return p+1
    else: raise Exception("mimetic basis function type wrong......")

# %%
def _bf_value( basis, x):
    poly_type, p = basis
    if poly_type == 'lobatto_node': return mbfv(x, p, 'LobN')
    elif poly_type == 'lobatto_edge': return mbfv(x, p, "LobE")
    elif poly_type == 'gauss_node': return mbfv(x, p, "GauN")
    elif poly_type == 'gauss_edge': return mbfv(x, p, "GauE")
    elif poly_type == 'ext_gauss_node': return mbfv(x, p, "etGN")
    elif poly_type == 'ext_gauss_edge': return mbfv(x, p, "etGE")
    else: raise Exception("mimetic basis function type wrong......")

# %%
def integral0d_(metric, basis_1, Quad = None):
    """
    #SUMMARY: Integrate "metric * basis_1" on [-1, 1]
    #OUTPUTS: [0] 1d array
    """
    if isinstance(metric, types.FunctionType): pass
    elif isinstance(metric, int) or isinstance(metric, float):
        temp=metric
        def fun(x): return temp
        metric=fun
    else: raise Exception("metric type wrong, only accept function, int or float")

    sd1 = _size_check(basis_1)

    if Quad is None:
        QuadType, QuadOrder = 'gauss', np.int(np.ceil((sd1)/2 + 1))
    else:
        QuadType, QuadOrder = Quad

    if   QuadType == 'gauss': Qnodes, weights = gauss_quad(QuadOrder)
    elif QuadType == 'lobatto': Qnodes, weights = lobatto_quad(QuadOrder)
    elif QuadType == 'extended_gauss': Qnodes, weights = extended_gauss_quad(QuadOrder)
    else: raise Exception("Quad Type should be gauss, lobatto or extended_gauss.......")

    basis_1 = _bf_value( basis_1, Qnodes)

    metric = metric(Qnodes)
    if np.size(metric) == 1: metric = metric * np.ones((np.size(Qnodes)))
    IntValue = np.einsum('ik,k,k->i', basis_1, metric, weights)
    
    return IntValue

# %%
def integral1d_(metric, basis_1, basis_2, Quad = None):
    """
    #SUMMARY: Integrate "metric * basis_1 * basis_2" on [-1, 1]
    #OUTPUTS: [0] 2d array: basis_1 -> 1st axis
                            basis_2 -> 2nd axis
    """
    if isinstance(metric, types.FunctionType): pass
    elif isinstance(metric, int) or isinstance(metric, float):
        temp=metric
        def fun(x): return temp
        metric=fun
    else: raise Exception("metric type wrong, only accept function, int or float")

    sd1 = _size_check(basis_1)
    sd2 = _size_check(basis_2)

    if Quad is None:
        QuadType, QuadOrder = 'gauss', np.int(np.ceil((sd1 + sd2)/2 + 1))
    else:
        QuadType, QuadOrder = Quad

    if   QuadType == 'gauss': Qnodes, weights = gauss_quad(QuadOrder)
    elif QuadType == 'lobatto': Qnodes, weights = lobatto_quad(QuadOrder)
    elif QuadType == 'extended_gauss': Qnodes, weights = extended_gauss_quad(QuadOrder)
    else: raise Exception("Quad Type should be gauss, lobatto or extended_gauss.......")

    basis_1 = _bf_value( basis_1, Qnodes)
    basis_2 = _bf_value( basis_2, Qnodes)

    metric = metric(Qnodes)

    if np.size(metric) == 1: metric = metric * np.ones((np.size(Qnodes)))
    IntValue = np.einsum('ik,jk,k,k->ij', basis_1, basis_2, metric, weights)
    
    return IntValue

# %%
if __name__ == '__main__':
    IntValue1 = integral1d_(1, ('lobatto_edge',3), ('gauss_node',5))
    print(IntValue1)

    IntValue0 = integral0d_(1, ('lobatto_edge',5))
    print(IntValue0)