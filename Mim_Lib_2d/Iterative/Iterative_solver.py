# -*- coding: utf-8 -*-
"""
Hybrid dual finite element method on Possion
#SOLVER Version 1.0
@author: Yi Zhang. Created on Sat Aug  5 11:44:19 2017
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
import numpy as np
import time
from assemble import assemble
from mesh_crazy import CrazyMesh
from scipy import sparse

from forms import form
from d import d
import functionals

import matplotlib.pyplot as plt
from scipy.sparse.linalg import inv
import scipy.io
import scipy


plt.rcParams['image.cmap'] = 'YlGnBu'
plt.rc('text', usetex=False)
font = {'fontname': 'Times New Roman',
        'color': 'k',
        'weight': 'normal',
        'size': 14}


# # %% define the exact solution
# def p0func(x, y):
#     return np.cos(3 * np.pi * x) * np.cos(4 * np.pi * y)
#
#
# def u(x, y):
#     return -3 * np.pi * np.sin(3 * np.pi * x) * np.cos(4 * np.pi * y)
#
#
# def v(x, y):
#     return -4 * np.pi * np.cos(3 * np.pi * x) * np.sin(4 * np.pi * y)
#
#
# def f2func(x, y):
#     return -25 * np.pi ** 2 * np.cos(3 * np.pi * x) * np.cos(4 * np.pi * y)


def p0func(x, y):
    return np.cos(3 * np.pi * x * y)


def u(x, y):
    return -3 * np.pi * y * np.sin(3 * np.pi * x * y)


def v(x, y):
    return -3 * np.pi * x * np.sin(3 * np.pi * x * y)


def f2func(x, y):
    return -9 * np.pi ** 2 * (x ** 2 + y ** 2) * np.cos(3 * np.pi * x * y)


def solver(p, n, c, precond=None, iterative=True):
    print("\n\n\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    print("\n\n===p,n,c===\n\n", p, n, c)
    # if n > 1:
    #     ele_pnt_x = functionals.lobatto_quad(n)[0]
    #     ele_pnt_y = functionals.lobatto_quad(n)[0]
    #
    #     ele_spa_x = [ele_pnt_x[i + 1] - ele_pnt_x[i] for i in range(0, n)]
    #     ele_spa_y = [ele_pnt_y[i + 1] - ele_pnt_y[i] for i in range(0, n)]
    #
    #     Element_Spacing = [ele_spa_x, ele_spa_y]
    # else:
    Element_Spacing = None

    # xi = np.linspace(-1, 1, 10)
    # xi, eta = np.meshgrid(xi, xi)
    # x, y = mesh.mapping(xi, eta)

    mesh = CrazyMesh(elements_layout=n, curvature=c, bounds_domain=((0, 2), (0, 2)), element_spacing=Element_Spacing)
    numbering_scheme = 'general'
    u1 = form(mesh, '1-lobatto', p, is_inner=False, numbering_scheme=numbering_scheme, separated_dof=True)
    f2 = form(mesh, '2-lobatto', p, is_inner=False, numbering_scheme=numbering_scheme)
    p0 = form(mesh, '0-gauss', p, is_inner=True, numbering_scheme=numbering_scheme)
    q0 = form(mesh, '0-gauss_tr', p, is_inner=True, numbering_scheme=numbering_scheme)

    # Obtain cochain of q0 and f2
    q0.discretize(p0func)
    f2.discretize(f2func)

    # set matrices for LHS
    M = u1.M()[1]
    E21 = d(u1)[0]
    W = f2.wedged(p0)
    Wb = q0.wedged(u1)

    #
    LHS11 = M
    LHS12 = assemble(W.dot(E21).T, u1.dof_map, p0.dof_map)
    LHS13 = assemble(Wb, u1.dof_map, q0.dof_map)
    LHS21 = LHS12.T
    LHS31 = LHS13.T

    LHS = sparse.bmat([[LHS11, LHS12, LHS13],
                       [LHS21, None, None]])

    LHS3 = sparse.lil_matrix(
        sparse.hstack((LHS31, sparse.csr_matrix((q0.num_dof, p0.num_dof)), sparse.csr_matrix((q0.num_dof, q0.num_dof)))))

    #
    rhs1 = np.zeros(u1.num_dof)
    rhs2 = assemble(W, f2.dof_map, p0.dof_map).dot(f2.cochain.reshape((f2.num_dof, 1))).ravel()
    rhs3 = np.zeros(q0.num_dof)

    # Boundary condition
    BC = q0.dof_map_boundary
    for row in BC:
        LHS3[row, :] = 0
        LHS3[row, u1.num_dof + p0.num_dof + np.array(row)] = 1
        rhs3[row] = q0.cochain[row]

    LHS = sparse.vstack((LHS, LHS3)).tolil()
    rhs = np.concatenate((rhs1, rhs2, rhs3))

    # If pre-conditioner is applied
    if precond is True:

        K = sparse.csc_matrix(LHS11)

        B = sparse.vstack((LHS21, LHS31))

        C = LHS[u1.num_dof:, u1.num_dof:]

        tic = time.time()

        pre_K = sparse.csc_matrix(np.diag(np.ones(shape=np.shape(K.diagonal())) / K.diagonal()))

        preconditioner_schur = (C + B * pre_K * B.transpose()).tocsc()

        # preconditioner_superLU = scipy.sparse.linalg.splu(preconditioner_schur, diag_pivot_thresh=0.0)

        # drop_tol=0.0, fill_factor=1000, diag_pivot_thresh=0)

        preconditioner_phi2 = inv(preconditioner_schur)  # preconditioner_superLU.L * preconditioner_superLU.U)

        # plt.matshow(preconditioner_phi2.todense())
        # plt.colorbar()
        # plt.show()

        KK = sparse.csc_matrix(preconditioner_phi2) * (C + B * inv(K) * B.transpose())

        KK_rhs = preconditioner_phi2 * (rhs[u1.num_dof:] + B * inv(K) * rhs[:u1.num_dof])

        toc = time.time()

        print("======time1:======", toc - tic)

        condKK = np.linalg.cond(KK.todense())
        print("\n======Condition number:======\n", condKK)

        RES2, t, iter_steps = Solver(KK.tocsc(), KK_rhs, iter_opt=iterative)

        print("\nThe assignment is done. This cost you in total: " + repr(t) + " seconds, " + repr(iter_steps) + " steps.")
        print('\nI will be at your service for the next assignment. \n\n')
        p0.cochain = RES2[:p0.num_dof].reshape(p0.num_dof)

    else:

        print("--------------------------------------------------------")
        print("------ solving the square sparse system......")
        print("\n======LHS shape:======\n\n", np.shape(LHS))
        # cond_LHS = np.linalg.cond(LHS.todense())
        # print("\n======Condition number:======\n", cond_LHS)

        Res, t, iter_steps = Solver(LHS.tocsc(), rhs, iter_opt=iterative)

        print("\nThe assignment is done. This cost you in total: " + repr(t) + " seconds, " + repr(iter_steps) + " steps.")
        print('\nI will be at your service for the next assignment. \n\n')
        p0.cochain = Res[u1.num_dof:u1.num_dof + p0.num_dof].reshape(p0.num_dof)

    # Reconstruction

    # u1.cochain = Res[:u1.num_dof].reshape(u1.num_dof)
    # p0.cochain = Res[u1.num_dof:u1.num_dof + p0.num_dof].reshape(p0.num_dof)

    # plot u results

    # mesh.plot_mesh(internal_mesh_type=('gauss', (p-1, p-1)))
    # u1.plot_self(plot_type='quiver')
    # u1.plot_self(plot_type='')

    # analytic solution

    # nn = 1000
    # xi_nn = 2. * np.arange(0, nn + 1) / 1000
    # [x_nn, y_nn] = np.meshgrid(xi_nn, xi_nn)
    # p0_nn = np.cos(3 * np.pi * x_nn * y_nn)
    # # p0_nn = -25 * np.pi ** 2 * np.cos(3 * np.pi * x_nn) * np.cos(4 * np.pi * y_nn)
    # plt.figure()
    # plt.contourf(x_nn, y_nn, p0_nn)
    # plt.colorbar()
    # plt.show()

    # plot p0 result
    p0.plot_self()

    # L2_error
    # u1_error = u1.L2_error((u, v))[0]
    p0_error = p0.L2_error(p0func)[0]
    # print('u1_error=', u1_error)
    print('p0_error=', p0_error)

    return p0_error, t


def Solver(A, b, iter_opt=False):
    if iter_opt is True:
        num_iter = 0

        def CallBack(xk):
            nonlocal num_iter
            num_iter += 1

        print("I'll start iterate......")
        print("\n======LHS shape:======\n\n", np.shape(A))
        t1 = time.time()
        (RES, warn_non_convergence) = sparse.linalg.cg(A, b, tol=1e-8, callback=CallBack, maxiter=20000)

        if warn_non_convergence:
            print("Did not converge")

        RES = np.array(RES)

        t2 = time.time()
        t = t2 - t1

        return RES, t, num_iter

    elif iter_opt is False:
        print("--------------------------------------------------------")
        print("------ solving the square sparse system......")
        print("\n======LHS shape:======\n\n", np.shape(A))
        t1 = time.time()
        RES = sparse.linalg.spsolve(A, b)
        t2 = time.time()
        RES = np.array(RES)

        return RES, t2-t1, 0
    else:
        raise Exception('iterative option should only be true or false')

# %%
if __name__ == "__main__":
    i = 0
    for p in [9]:
        for c in [0]:
            for n in [2]:
                Res_i = solver(p, n, c, precond=False, iterative=False)
                info = "p0_error, t"
                if i == 0:
                    Res = np.array([p, n, c, *Res_i])
                else:
                    Res = np.vstack((Res, np.array([p, n, c, *Res_i])))
                i += 1

description = "SOVER_V1_TIME_COSTS_MEASURE"
np.save('SOVER_V1_TIME_COSTS_MEASURE_lp', (Res, info, description))
