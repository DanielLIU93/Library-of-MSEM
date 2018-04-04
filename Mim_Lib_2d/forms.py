# -*- coding: utf-8 -*-
"""
We use this function to call all coded forms

@author: Yi Zhang. Created on Tue Aug 29 10:45:33 2017
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""


import dictionaries
from mesh_crazy import CrazyMesh
import matplotlib.pyplot as plt
forms_dict = dictionaries.forms

""""""


def form(mesh, what_form, p=1, is_inner=True, separated_dof=True, the_form=None, trace_form=None, numbering_scheme=None,
         name=None, info=None):
    """wrap of all forms"""
    assert what_form in forms_dict, " <FORM> : {} is not in the dictionary".format(what_form)
    (form_order, form_type) = what_form.split('-')

    #%% LOBATTO
    if form_type == 'lobatto':
        return forms_dict[what_form](mesh, p, is_inner=is_inner, separated_dof=separated_dof,
                                     numbering_scheme=numbering_scheme, name=name, info=info)

    #%% GAUSS
    elif form_type == 'gauss':
        return forms_dict[what_form](mesh, p, is_inner=is_inner)

    #%% EXT_GAUSS
    elif form_type == 'ext_gauss':
        if form_order == '0':
            return forms_dict[what_form](mesh, p, is_inner=is_inner, the_form=the_form, trace_form=trace_form,
                                         numbering_scheme=numbering_scheme, name=name, info=info)
        elif form_order == '1':
            return forms_dict[what_form](mesh, p, is_inner=is_inner, trace_form=trace_form,
                                         numbering_scheme=numbering_scheme, name=name, info=info)
        elif form_order == '2':
            return forms_dict[what_form](mesh, p, is_inner=is_inner,
                                         numbering_scheme=numbering_scheme, name=name, info=info)
        else:
            raise Exception(" <FORM> : " + what_form + ' is not coded')

    #%% TRACE_GAUSS
    elif form_type in ('gauss_tr', 'ext_gauss_tr'):
        return forms_dict[what_form](mesh, p, is_inner=is_inner,
                                     numbering_scheme=numbering_scheme, name=name, info=info)

    #%%
    else:
        raise Exception(' <FORM> : form type: ' + what_form + ' is not coded')

# %% MAIN
if __name__ == '__main__':
    p = (2, 2)
    n = (3, 3)
    mesh = CrazyMesh(elements_layout=n, bounds_domain=((0, 1), (0, 1)))
    f = form(mesh, '0-gauss', p=p)
    mesh.plot_mesh()
    # f.plot_self()