# -*- coding: utf-8 -*-
"""
Here I store all assistant functions about polynomials
"""
import numpy as np
from functools import partial
from scipy.special import legendre

# %%
def lobatto_quad(p):
    """Gauss Lobatto quadrature.

    Args:
        p (int) = order of quadrature

    Returns:
        nodal_pts (np.array) = nodal points of quadrature
        w (np.array) = correspodent weights of the quarature.
    """
    # nodes
    x_0 = np.cos(np.arange(1, p) / p * np.pi)
    # temp = np.arange(1, p) / p * np.pi
    # x_0 = np.zeros(p-1)
    # for i, temp_pnt in enumerate(temp):
    #     if temp_pnt <= 0.5 * np.pi:
    #         x_0[i] = 1 - np.sin(temp_pnt)
    #     else:
    #         x_0[i] = np.sin(temp_pnt) - 1

    nodal_pts = np.zeros((p + 1))
    # final and initial point
    nodal_pts[0] = 1
    nodal_pts[-1] = -1
    # Newton method for root finding
    for i, ch_pt in enumerate(x_0):
        leg_p = partial(_legendre_prime_lobatto, n=p)
        leg_pp = partial(_legendre_double_prime, n=p)
        nodal_pts[i + 1] = _newton_method(leg_p, leg_pp, ch_pt, 100)

    # weights
    weights = 2 / (p * (p + 1) * (legendre(p)(nodal_pts))**2)

    return nodal_pts[::-1], weights

# %%
def gauss_quad(p):
    # Chebychev pts as inital guess
    x_0 = np.cos(np.arange(1, p + 1) / (p + 1) * np.pi)
    nodal_pts = np.empty(p)
    for i, ch_pt in enumerate(x_0):
        leg = legendre(p)
        leg_p = partial(_legendre_prime, n=p)
        nodal_pts[i] = _newton_method(leg, leg_p, ch_pt, 100)

    weights = 2 / (p * legendre(p - 1)(nodal_pts)
                   * _legendre_prime(nodal_pts, p))
    return nodal_pts[::-1], weights


# %%
def extended_gauss_quad(p):
    nodes, weights = gauss_quad(p)
    ext_nodes = np.ones((p + 2))
    ext_nodes[0] = -1
    ext_nodes[1:-1] = nodes
    ext_weights = np.zeros(p + 2)
    ext_weights[1:-1] = weights
    return ext_nodes, ext_weights


# %%
def lagrange_basis(nodes, x=None):
    if x is None:
        x = nodes
    p = np.size(nodes)
    basis = np.ones((p, np.size(x)))
    # lagrange basis functions
    for i in range(p):
        for j in range(p):
            if i != j:
                basis[i, :] *= (x - nodes[j]) / (nodes[i] - nodes[j])
    return basis


# %%
def edge_basis(nodes, x=None):
    """Return the edge polynomials."""
    if x is None:
        x = nodes
    p = np.size(nodes) - 1
    derivatives_poly = _derivative_poly(p, nodes, x)
    edge_poly = np.zeros((p, np.size(x)))
    for i in range(p):
        for j in range(i + 1):
            edge_poly[i] -= derivatives_poly[j, :]
    return edge_poly


# %%
def _derivative_poly_nodes(p, nodes):
    """For computation of the derivative at the nodes a more efficient and accurate formula can
       be used, see [1]:

                 | \frac{c_{k}}{c_{j}}\frac{1}{x_{k}-x_{j}},          k \neq j
                 |
       d_{kj} = <
                 | \sum_{l=1,l\neq k}^{p+1}\frac{1}{x_{k}-x_{l}},     k = j
                 |

                 with
        c_{k} = \prod_{l=1,l\neq k}^{p+1} (x_{k}-x_{l}).

    Args:
        p (int) = degree of polynomial
        type_poly (string) = 'Lobatto', 'Gauss', 'Extended Gauss'
    [1] Costa, B., Don, W. S.: On the computation of high order
      pseudo-spectral derivatives, Applied Numerical Mathematics, vol.33
       (1-4), pp. 151-159
        """
    # compute distances between the nodes
    xi_xj = nodes.reshape(p + 1, 1) - nodes.reshape(1, p + 1)
    # diagonals to one
    xi_xj[np.diag_indices(p + 1)] = 1
    # compute (ci's)
    c_i = np.prod(xi_xj, axis=1)
    # compute ci/cj = ci_cj(i,j)
    c_i_div_cj = np.transpose(c_i.reshape(1, p + 1) / c_i.reshape(p + 1, 1))
    # result formula
    derivative = c_i_div_cj / xi_xj
    # put the diagonals equal to zeros
    derivative[np.diag_indices(p + 1)] = 0
    # compute the diagonal values enforning sum over rows = 0
    derivative[np.diag_indices(p + 1)] = -np.sum(derivative, axis=1)

    return derivative


# %%
def _derivative_poly(p, nodes, x):
    """Return the derivatives of the polynomials in the domain x."""
    nodal_derivative = _derivative_poly_nodes(p, nodes)
    polynomials = lagrange_basis(nodes, x)
    return np.transpose(nodal_derivative) @ polynomials


# %%
def _legendre_prime(x, n):
    """Calculate first derivative of the nth Legendre Polynomial recursively.

    Args:
        x (float,np.array) = domain.
        n (int) = degree of Legendre polynomial (L_n).
    Return:
        legendre_p (np.array) = value first derivative of L_n.
    """
    # P'_n+1 = (2n+1) P_n + P'_n-1
    # where P'_0 = 0 and P'_1 = 1
    # source: http://www.physicspages.com/2011/03/12/legendre-polynomials-recurrence-relations-ode/
    if n == 0:
        if isinstance(x, np.ndarray):
            return np.zeros(len(x))
        elif isinstance(x, (int, float)):
            return 0
    if n == 1:
        if isinstance(x, np.ndarray):
            return np.ones(len(x))
        elif isinstance(x, (int, float)):
            return 1
    legendre_p = (n * legendre(n - 1)(x) - n * x * legendre(n)(x))/(1-x**2)
    return legendre_p


# %%
def _legendre_prime_lobatto(x,n):
    return (1-x**2)**2*_legendre_prime(x,n)


# %%
def _legendre_double_prime(x, n):
    """Calculate second derivative legendre polynomial recursively.

    Args:
        x (float,np.array) = domain.
        n (int) = degree of Legendre polynomial (L_n).
    Return:
        legendre_pp (np.array) = value second derivative of L_n.
    """
    legendre_pp = 2 * x * _legendre_prime(x, n) - n * (n + 1) * legendre(n)(x)
    return legendre_pp * (1 - x ** 2)


# %%
def _newton_method(f, dfdx, x_0, n_max, min_error=np.finfo(float).eps * 10):
    """Newton method for rootfinding.

    It garantees quadratic convergence given f'(root) != 0 and abs(f'(ξ)) < 1
    over the domain considered.

    Args:
        f (obj func) = function
        dfdx (obj func) = derivative of f
        x_0 (float) = starting point
        n_max (int) = max number of iterations
        min_error (float) = min allowed error

    Returns:
        x[-1] (float) = root of f
        x (np.array) = history of convergence
    """
    x = [x_0]
    for i in range(n_max - 1):
        x.append(x[i] - f(x[i]) / dfdx(x[i]))
        if abs(x[i + 1] - x[i]) < min_error:
            return x[-1]

    print('WARNING : Newton did not converge to machine precision \nRelative error : ', x[-1] - x[-2])
    return x[-1]