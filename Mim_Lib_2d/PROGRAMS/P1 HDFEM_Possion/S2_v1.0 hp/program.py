# -*- coding: utf-8 -*-
"""
Hybrid dual finite element method on Possion, hp version
#SOLVER Version 1.0
@author: Yi Zhang. Created on Sat Aug  5 11:44:19 2017
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
import time
import numpy as np
from mesh_crazy import CrazyMesh
from tqdm import tqdm
from numpy.linalg import inv
from assemble import assemble
from scipy import sparse
from forms import form
from d import d

import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'bwr'
plt.rc('text', usetex=False)
font = {'fontname': 'Times New Roman',
        'color': 'k',
        'weight': 'normal',
        'size': 12}


# %% define the exact solution
# def p0func(x, y): return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
#
#
# def u(x, y): return 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
#
#
# def v(x, y): return 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
#
#
# def f2func(x, y): return -8 * np.pi ** 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

def p0func(x, y):
    return np.cos(3 * np.pi * x) * np.cos(4 * np.pi * y)


def u(x, y):
    return -3 * np.pi * np.sin(3 * np.pi * x) * np.cos(4 * np.pi * y)


def v(x, y):
    return -4 * np.pi * np.cos(3 * np.pi * x) * np.sin(4 * np.pi * y)


def f2func(x, y):
    return -25 * np.pi ** 2 * np.cos(3 * np.pi * x) * np.cos(4 * np.pi * y)

# def p0func(x, y):
#     return np.cos(3 * np.pi * x * y)
#
#
# def u(x, y):
#     return -3 * np.pi * y * np.sin(3 * np.pi * x * y)
#
#
# def v(x, y):
#     return -3 * np.pi * x * np.sin(3 * np.pi * x * y)
#
#
# def f2func(x, y):
#     return -9 * np.pi ** 2 * (x + y) * np.sin(3 * np.pi * x * y)


# %% THE SOLVER BODY
def solver(p, n, c):
    p = (p, p)
    n = (n, n)
    print("\n\n\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("p,n,c=", p, n, c)
    mesh = CrazyMesh(elements_layout=n, curvature=c, bounds_domain=((0, 2), (0, 2)))
    u1 = form(mesh, '1-lobatto', p, is_inner=False, numbering_scheme='general')
    f2 = form(mesh, '2-lobatto', p, is_inner=False, numbering_scheme='general')
    p0 = form(mesh, '0-gauss', p, is_inner=True, numbering_scheme='general')
    q0 = form(mesh, '0-gauss_tr', p, is_inner=True, numbering_scheme='general')
    phi = form(mesh, '0-ext_gauss', p, is_inner=True, numbering_scheme='general')

    #  KNOWN VALUES
    q0.discretize(p0func)
    f2.discretize(f2func)

    # MASS MATRIX AND INVERT
    M = u1.M()[0]
    E21 = d(u1)[0]
    W = f2.wedged(p0)

    a21 = W.dot(E21)

    M = np.rollaxis(M, 2, 0)
    a21 = np.rollaxis(np.repeat(a21[:, :, np.newaxis], mesh.num_elements, axis=2), 2, 0)
    a12 = np.transpose(a21, (0, 2, 1))

    a1 = np.concatenate((M, a12), axis=2)
    a2 = np.concatenate((a21, np.zeros(shape=(mesh.num_elements, p0.num_basis, p0.num_basis))), axis=2)
    a = np.concatenate((a1, a2), axis=1)
    print("\n--------------------------------------------------------")
    print("M shape:", np.shape(a))
    print("------ invert the mass matrix locally: ......")
    t1 = time.time()
    inva = inv(a)
    t2 = time.time()
    t_inv_M = t2 - t1
    print("------DONE,costs:", t_inv_M)

    # %% the_global_global_numbering
    ggn = np.array([int(i) for i in range(mesh.num_elements * (p0.num_basis + u1.num_basis))]).reshape(
        (mesh.num_elements, p0.num_basis + u1.num_basis), order="C")

    invA = assemble(np.rollaxis(inva, 0, 3), ggn, ggn)

    u1_dof_map = ggn[:, :u1.num_basis]

    # %% THE BOUNDARY WEDGE MATRIX
    Wb = q0.wedged(u1)
    CT = sparse.vstack((assemble(Wb, u1_dof_map, q0.dof_map), sparse.csc_matrix((p0.num_basis, q0.num_dof))))
    C = CT.T

    # %% alpha
    f2.cochain = assemble(W, f2.dof_map, p0.dof_map).dot(f2.cochain.reshape((f2.num_dof, 1))).reshape((f2.num_dof))
    alpha = np.concatenate((np.zeros(shape=(u1.num_basis, mesh.num_elements)), f2.cochain_local), axis=0).reshape(
        (u1.num_dof + f2.num_dof, 1), order='F')

    # %% beta, rhs, BC
    # beta = np.zeros(shape=(q0.num_dof, 1))
    # D = np.zeros( (q0.num_dof,q0.num_dof) )
    #
    # BC = q0.dof_map_boundary
    # for i,row in enumerate(BC):
    #    for j in row:
    #        C[ j, :] = 0
    #        D[ j, j] = 1
    #        beta[ j] = q0.cochain[j]

    # lhs =sparse.lil_matrix(C.dot(invA).dot(CT)-D)
    # rhs = C.dot(invA).dot(alpha) - beta

    # %% CONSTRUCT lhs and rhs
    print("\n--------------------------------------------------------")
    print("construct the lhs and rhs:")
    t3 = time.time()
    lhs = C.dot(invA).dot(CT)
    rhs = C.dot(invA).dot(alpha)
    t4 = time.time()
    t_construct_lhs_rhs = t4 - t3
    print("------ DONE,costs:", t_construct_lhs_rhs)

    # %% BC
    print("\n--------------------------------------------------------")
    print("convert into lil sparse......")
    lhs = sparse.lil_matrix(lhs)
    print("get dof BC......")
    BC = q0.dof_map_boundary
    print("impose the BC:")
    t1 = time.time()
    for _, row in tqdm(enumerate(BC)):
        lhs[row, :] = 0
        lhs[row, row] = 1
        rhs[row] = q0.cochain[row].reshape((np.size(row), 1))
    t2 = time.time()
    print("------ DONE,costs:", t2 - t1)

    # %% CONDITION NUMBER
    #    print("\n--------------------------------------------------------")
    #    print("compute the condition number of lhs:")
    #    t1 = time.time()
    #    cond_num = np.linalg.cond(lhs.toarray())
    #    t2 = time.time()
    #    t_cond_num = t2 - t1
    #    print('cond_num = ', cond_num)
    #    print("------ DONE,costs:", t_cond_num)

    # %% SOLVER SPARESE SYSTEM
    print("\n--------------------------------------------------------")
    print("lhs shape:", np.shape(lhs))
    #    
    lhs = sparse.csr_matrix(lhs)
    print("------ solve the square sparse system locally......")
    t5 = time.time()
    q_Res = sparse.linalg.spsolve(lhs, rhs)
    t6 = time.time()
    t_solve_q0 = t6 - t5
    print("------ DONE,costs:", t_solve_q0)

    # %% RENEW q0 or phi
    q0.cochain = q_Res

    # %% RHS
    a13 = np.rollaxis(np.repeat(Wb[:, :, np.newaxis], mesh.num_elements, axis=2), 2, 0)
    a1 = np.concatenate((M, a12, a13), axis=2)
    a2 = np.concatenate((a21,
                         np.zeros(shape=(mesh.num_elements, p0.num_basis, p0.num_basis)),
                         np.zeros(shape=(mesh.num_elements, p0.num_basis, q0.num_basis)),
                         ), axis=2)

    a3 = np.concatenate((np.zeros(shape=(mesh.num_elements, q0.num_basis, u1.num_basis)),
                         np.zeros(shape=(mesh.num_elements, q0.num_basis, p0.num_basis)),
                         np.repeat(np.eye(q0.num_basis, dtype=int)[np.newaxis, :, :], mesh.num_elements, axis=0)
                         ), axis=2)

    LHS = np.concatenate((a1, a2, a3), axis=1)
    u1.cochain = np.zeros((u1.num_dof))
    RHS = np.concatenate((u1.cochain_local.T, f2.cochain_local.T, q0.cochain_local.T), axis=1)

    # %% SOVER u and p LOCALLY
    print("\n--------------------------------------------------------")
    print("LHS shape:", np.shape(LHS))
    #    
    print("------ solve the square sub-systems......")
    t7 = time.time()
    Res = np.linalg.solve(LHS, RHS)
    t8 = time.time()
    t_solve_local = t8 - t7
    print("------ DONE, costs:", t_solve_local)

    # %% SPLIT THE SOLUTION
    u1.cochain_local = Res[:, :u1.num_basis].T
    p0.cochain_local = Res[:, u1.num_basis:-q0.num_basis].T

    #    mesh.plot_mesh()
    # u1.plot_self()
    p0.plot_self()

    # %% COMPUTE DP AND DU
    phi.compose(p0, q0)
    du = d(u1)[2]

    # %% ERRORS AND TIME EFFICIENCY
    print("\n--------------------------------------------------------")
    print("Solving summary:")
    t = t_inv_M + t_construct_lhs_rhs + t_solve_q0
    print('solving costs:', t)

    u1_L2_error = u1.L2_error((u, v))[0]
    print('u1_error=', u1_L2_error)

    p0_L2_error = p0.L2_error(p0func)[0]
    print('p0_error=', p0_L2_error)

    du_L2_error = du.L2_error(f2func)[0]
    print('du_error=', du_L2_error)

    u_Hdiv_error = np.sqrt(u1_L2_error ** 2 + du_L2_error ** 2)
    print('u_Hdiv_error=', u_Hdiv_error)

    f2.discretize(f2func)
    dumf = du - f2
    dumf_L2_error = dumf.L2_error(func=lambda x, y: 0 * x * y)[0]

    print('dumf_L2_error=', dumf_L2_error)

    return p0_L2_error, u1_L2_error, du_L2_error, u_Hdiv_error, dumf_L2_error, t


# %% THE MAIN PROGRAM
if __name__ == "__main__":
    i = 0
    for p in [25]:
        for c in [0.0]:
            for n in [1]:
                Res_i = solver(p, n, c)

                info = "p0_L2_error, u1_L2_error, du_L2_error, u_Hdiv_error, dumf_L2_error, t"
                if i == 0:
                    Res = np.array([p, n, c, *Res_i])
                else:
                    Res = np.vstack((Res, np.array([p, n, c, *Res_i])))
                i += 1

    description = "SOVER_V2_hp_TIME_COST_MEASURE_hp"

    #    np.save('SOVER_V2_hp_TIME_COST_MEASURE_hp', (Res, info, description))
