# -*- coding: utf-8 -*-
"""
A wrap of all numbering schemes

@author: Yi Zhang （张仪）, Created on Thu Oct 26 17:52:58 2017
    Aerodynamics
    Faculty of Aerospace Engineering
    TU Delft
"""
from numbering_general import general
from numbering_symmetric import symmetric


def numbering(form, numbering_scheme=None):

    if numbering_scheme is "general":
        return general(form)

    elif numbering_scheme is None:
        return general(form)

    elif numbering_scheme is "symmetric1":
        return symmetric(form)

    else:
        raise Exception(" <NUMB> : numbering scheme: {} not coded".format(numbering_scheme))