# -*- coding: utf-8 -*-
"""
A GENERAL NUMBERING using information from elements_map

@author: Yi Zhang. Created in 2017
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
import numpy as np


# %% THE FUNCTION BODY
def symmetric(form):
    version = 1.0
    num_elements = form.mesh.num_elements
    px, py = form.p

    # %% GAUSS 0 FORM
    if form.__class__.__name__ == 'Gauss0form':
        """ SUMMARY """
        return np.array([int(i) for i in range(num_elements*px*py)]).reshape(
                num_elements, px*py, order='C')

    # %% GAUSS 1 FORM
    elif form.__class__.__name__ == 'Gauss1form':
        """ SUMMARY """
        return np.array([int(i) for i in range(num_elements*form.num_basis)]).reshape(
                num_elements, form.num_basis, order='C')

    # %% GAUSS 2 FORM
    elif form.__class__.__name__ == 'Gauss2form':
        """ SUMMARY """
        return np.array([int(i) for i in range(num_elements*form.num_basis)]).reshape(
                num_elements, form.num_basis, order='C')

    # %% GAUSS_LOBATTO 0 FORM
    elif form.__class__.__name__ == 'Lobatto0form' and form.separated_dof is True:
        """ SUMMARY """
        global_numbering = np.array([int(i) for i in range(num_elements * (px+1) * (py+1))]).reshape(
                num_elements, (px+1) * (py+1), order='C')

        elements_map = form.mesh.elements_map

        numbering_boundaries = ()
        for i in range(form.mesh.R.num_boundaries_sections):
            numbering_boundaries += ([],)

        local_index = {'0': (0, (px+1)*(py+1), (py+1)),
                       '1': (py, (px+1)*(py+1), (py+1)),
                       '2': (0, py+1, 1),
                       '3': (px*(py+1), (px+1)*(py+1), 1)
                       }

        for i, _map in enumerate(elements_map):
            for j, position in enumerate(_map):
                if isinstance(position, str):
                    numbering_boundaries[int(position)].extend(
                            global_numbering[i, local_index[str(j)][0]:local_index[str(j)][1]:local_index[str(j)][2]]
                            .tolist())

        return global_numbering, numbering_boundaries

    # %% GAUSS_LOBATTO 0 FORM CONTINUOUS NUMBERING
    elif form.__class__.__name__ == 'Lobatto0form' and form.separated_dof is False:
        """ SUMMARY """
        # TODO:
        raise Exception("Lobatto0form with not separated dof is not coded yet")

    # %% GAUSS_LOBATTO 1 FORM
    elif form.__class__.__name__ == 'Lobatto1form' and form.separated_dof is True:
        """ SUMMARY """
        global_numbering = np.array([int(i) for i in range(num_elements * (px*(py+1) + py*(px+1)))]).\
            reshape(num_elements, (px*(py+1) + py*(px+1)), order='C')

        elements_map = form.mesh.elements_map

        numbering_boundaries = ()
        for i in range(form.mesh.R.num_boundaries_sections):
            numbering_boundaries += ([],)

        local_index = {'0': (0, px*(py+1), (py+1)),
                       '1': (py, px*(py+1), (py+1)),
                       '2': (px*(py+1), px*(py+1)+py, 1),
                       '3': (px*(py+1)+py*px, px*(py+1)+py*(px+1), 1)
                       }

        for i, _map in enumerate(elements_map):
            for j, position in enumerate(_map):
                if isinstance(position, str):
                    numbering_boundaries[int(position)].\
                        extend(global_numbering[i, local_index[str(j)][0]:local_index[str(j)][1]:local_index[str(j)][2]]
                               .tolist())

        return global_numbering, numbering_boundaries

    # %% GAUSS_LOBATTO 1 FORM   CONTINUOUS NUMBERING
    elif form.__class__.__name__ == 'Lobatto1form' and form.separated_dof is False:
        """ SUMMARY """
        # TODO:
        raise Exception("Lobatto1form with not separated dof is not coded yet")

    # %% GAUSS_LOBATTO 2 FORM
    elif form.__class__.__name__ == 'Lobatto2form':
        """ SUMMARY """
        return np.array([int(i) for i in range(num_elements * px * py)]).reshape(
                num_elements, px*py, order='C')

    # %% EXTENDED GAUSS 1FORM (INTERNAL PART OF COURSE)
    elif form.__class__.__name__ == 'ExtGauss1form':
        """ SUMMARY """
        return np.array([int(i) for i in range(num_elements * (px*(py+1) + py*(px+1)))]).reshape(
                num_elements, (px*(py+1) + py*(px+1)), order='C')

    # %% EXTENDED GAUSS 2FORM
    elif form.__class__.__name__ == 'ExtGauss2form':
        """ SUMMARY """
        return np.array([int(i) for i in range(num_elements*(px+1)*(py+1))]).reshape(
                num_elements, ((px+1)*(py+1)), order='C')

    # %% TRACE GAUSS 0-FORM
    elif form.__class__.__name__ == 'TraceGauss0form':
        """
        #SUMMARY: Locally, we first numbering the south boundary of element.
                  Then the north, then the west, then the east

                  Then we go element by element following above rule, if already numbered, then just take it.
        """
        if form.mesh.elements_layout != (1, 1):
            assert form.mesh._lowest_numbering_version <= version, 'Mesh requires higher version numbering.py'
        
        elements_map = form.mesh.elements_map
        global_numbering = np.zeros(shape=(num_elements, 2*(px+py)), dtype=np.int64)
        numbering_boundaries = ()
        for i in range(form.mesh.R.num_boundaries_sections):
            numbering_boundaries += ([],)

        local_index = {'0': (0, px),
                       '1': (px, 2*px),
                       '2': (2*px, 2*px+py),
                       '3': (2*px+py, 2*px+2*py)
                       }

        num_points = {'0': px,
                      '1': px,
                      '2': py,
                      '3': py
                      }

        current_number = 0

        for i, _map in enumerate(elements_map):
            for j, position in enumerate(_map):
                if isinstance(position, str):  # on the domain boundary
                    global_numbering[i, local_index[str(j)][0]:local_index[str(j)][1]] = [
                            int(i) for i in range(current_number, current_number+num_points[str(j)])]

                    numbering_boundaries[int(position)].extend([
                            int(i) for i in range(current_number, current_number+num_points[str(j)])])
                    current_number += num_points[str(j)]

                else:  # another element
                    if np.abs(position) > i:  # not numbered yet
                        global_numbering[i, local_index[str(j)][0]:local_index[str(j)][1]] = [
                                int(i) for i in range(current_number, current_number+num_points[str(j)])]
                        current_number += num_points[str(j)]

                    elif np.abs(position) < i:
                        m = np.abs(position)
                        if i in elements_map[m]:
                            n = list(elements_map[m]).index(i)
                        elif -i in elements_map[m]:
                            n = list(elements_map[m]).index(-i)
                        else:
                            raise Exception("in Element No.{}, can not find element No.{} or {}".format(m, i, -i))

                        # self location: (i,j), target location: (m, n)
                        if elements_map[i][j] + elements_map[m][n] < 0:
                            global_numbering[i, local_index[str(j)][0]:local_index[str(j)][1]] = \
                                global_numbering[m, local_index[str(n)][0]:local_index[str(n)][1]][::-1]
                        else:
                            global_numbering[i, local_index[str(j)][0]:local_index[str(j)][1]] = \
                                global_numbering[m, local_index[str(n)][0]:local_index[str(n)][1]]
                    else:
                        raise Exception("Wrong: element_map[{}] contains element No{} itself".format(i, i))

        return global_numbering, numbering_boundaries

    # %% Not code part
    else:
        raise Exception("Not implemented yet for form:"+form.__class__.__name__)