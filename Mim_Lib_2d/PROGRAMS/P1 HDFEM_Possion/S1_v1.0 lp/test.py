import numpy as np
import time
from mesh_crazy import CrazyMesh
from assemble import assemble
from scipy import sparse

from forms import form
from d import d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio


def p0func(x, y):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def u(x, y):
    return 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)


def v(x, y):
    return 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


def f2func(x, y):
    return -8 * np.pi ** 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def single_element_solver(p: object, n: object = 1, c: object = 0.15, num_scheme=None) -> object:
    p = (p, p)  # order of the accuracy
    n = (n, n)  # element numbers
    print("\n\n\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    print("p,n,c=", p, n, c)
    mesh = CrazyMesh(elements_layout=n, curvature=c, bounds_domain=((0, 1), (0, 1)))
    u1 = form(mesh, '1-lobatto', p, is_inner=False, numbering_scheme=num_scheme)
    f2 = form(mesh, '2-lobatto', p, is_inner=False, numbering_scheme=num_scheme)
    p0 = form(mesh, '0-gauss', p, is_inner=True, numbering_scheme=num_scheme)
    q0 = form(mesh, '0-gauss_tr', p, is_inner=True, numbering_scheme=num_scheme)

    q0.discretize(p0func)
    f2.discretize(f2func)

    M = u1.M()[1]
    # M_print = M.todense()
    # print('\n\n===mass===\n\n', M_print)
    # plt.matshow(M_print)
    # plt.colorbar()
    # plt.set_cmap('YlGnBu')
    # plt.show()

    tE21 = d(u1)[0]
    W = f2.wedged(p0)
    B = q0.wedged(u1)

    if p[0] == 25:
        sio.savemat("Mass_matrix_25.mat", mdict={'M_25': M})
        sio.savemat('Incidence_matrix_25.mat', mdict={'E21_25': tE21})
        sio.savemat('Wedge_matrix_25.mat', mdict={'WedgeMat_25': W})
        sio.savemat('B_Wedge_25.mat', mdict={'BWedge_25': B})
        sio.savemat('q0_25', mdict={'q0_25': q0.dof_map})
        sio.savemat('f2_25', mdict={'f2_25': f2.dof_map})

    if p[0] == 5:
        sio.savemat("Mass_matrix_5.mat", mdict={'M_5': M})
        sio.savemat('Incidence_matrix_5.mat', mdict={'E21_5': tE21})
        sio.savemat('Wedge_matrix_5.mat', mdict={'WedgeMat_5': W})
        sio.savemat('B_Wedge_5.mat', mdict={'BWedge_5': B})
        sio.savemat('q0_5', mdict={'q0_5': q0.dof_map})
        sio.savemat('f2_5', mdict={'f2_5': f2.dof_map})

    if p[0] == 9:
        sio.savemat("Mass_matrix_9.mat", mdict={'M_9': M})
        sio.savemat('Incidence_matrix_9.mat', mdict={'E21_9': tE21})
        sio.savemat('Wedge_matrix_9.mat', mdict={'WedgeMat_9': W})
        sio.savemat('B_Wedge_9.mat', mdict={'BWedge_9': B})
        sio.savemat('q0_9', mdict={'q0_9': q0.dof_map})
        sio.savemat('f2_9', mdict={'f2_9': f2.dof_map})

    if p[0] == 13:
        sio.savemat("Mass_matrix_13_map.mat", mdict={'M_13': M})
        sio.savemat('Incidence_matrix_13.mat', mdict={'E21_13': tE21})
        sio.savemat('Wedge_matrix_13.mat', mdict={'WedgeMat_13': W})
        sio.savemat('B_Wedge_13.mat', mdict={'BWedge_13': B})
        sio.savemat('q0_13', mdict={'q0_13': q0.dof_map})
        sio.savemat('f2_13', mdict={'f2_13': f2.dof_map})

    if p[0] == 40:
        sio.savemat("Mass_matrix_45.mat", mdict={'M_45': M})
        sio.savemat('Incidence_matrix_45.mat', mdict={'E21_45': tE21})
        sio.savemat('Wedge_matrix_45.mat', mdict={'WedgeMat_45': W})
        sio.savemat('B_Wedge_45.mat', mdict={'BWedge_45': B})
        sio.savemat('q0_45', mdict={'q0_45': q0.dof_map})
        sio.savemat('f2_45', mdict={'f2_45': f2.dof_map})

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # X = np.arange(np.shape(M)[0])
    # Y = X
    # X, Y = np.meshgrid(X, Y)
    # Z = M.toarray
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

if __name__ == "__main__":

    single_element_solver(5, 1, 0, num_scheme='general')


