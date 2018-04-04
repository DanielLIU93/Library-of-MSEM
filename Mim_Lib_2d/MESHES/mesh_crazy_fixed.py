"""
Crazy mesh

@author: Created by Lorenzo and Varun [2017]
         Modified by Yi Zhang [2017]
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
import numpy as np
import warnings
from regions import Regions
from mesh_basic import BasicMesh
# import matplotlib as plt


# %% THE CRAZY MESH CLASS BODY
class CrazyMesh(BasicMesh):
    """
    CrazyMesh class generates a tensor product mesh. The mesh can be deformed
    through a non linear mapping.

    Args:
    ----
        dim :   int
                dim is the dimension of the manifold
                # currenly implemente just in 2D

        num_elements :  tuple (int, size : 2)
                        num_elements specifies the number of elements in the x
                        and y dimension.
                        example : num_elements = (i,j) implements i elements in x
                        and j elements in y direction in the mesh

        bounds_domain : tuple (float64, size : 4)
                        bounds_domain specifies the nodal coordinates at the corners
                        of the domain. The first entry is a tuple containing the
                        coordinates of the extreme points in the x dimension, second entry
                        the cooridinates in the y dimension
                        example : bounds_domain = ((a_x,b_x),(a_y,b_y)) where
                        b_x > a_x, b_y > a_y

        curvature :     float64 (optional)
                        specifies the curvature to be applied to the domain.
                        Typical range [0-0.3]

        nodes :         tuple()
    """
    # %% __INIT__ MESH
    def __init__(self, elements_layout, bounds_domain, curvature=0., info=None, name=None, pre_No=None):
        SW = (bounds_domain[0][0], bounds_domain[1][0])
        SE = (bounds_domain[0][1], bounds_domain[1][0])
        NW = (bounds_domain[0][0], bounds_domain[1][1])
        NE = (bounds_domain[0][1], bounds_domain[1][1])
        regions = {'R_crazy': (SW, SE, NW, NE), }
        region_type = {}
        boundaries_dict = {'South': ('0S',),
                           'North': ('0N',),
                           'West': ('0W',),
                           'East': ('0E',)}
        REGIONS = (regions, region_type, boundaries_dict)
        R = Regions(REGIONS)

        super().__init__(2, R, elements_layout, info, name, pre_No)

        if curvature > 0.316:
            warnings.warn(" <MESH> <CRAZY> : curvature={} > 0.316 (normally curvature in [0, 0.3]), grid may go outside domain somewhere!".format(curvature))
        self._curvature = curvature

        (self._x_bounds, self._y_bounds) = bounds_domain
        self._bounds_domain = bounds_domain

        # self.self_checker

    # %% PROPERTIES
    @property
    def curvature(self):
        return self._curvature

    # %%
    @property
    def x_bounds(self):
        return self._x_bounds

    @property
    def y_bounds(self):
        return self._y_bounds

    @property
    def bounds_domain(self):
        return self._bounds_domain

    # %% THE MAPING
    def mapping(self, xi, eta, element=None):
        if not isinstance(element, int):
            if element is None:
                element = np.arange(self.num_elements)

            x, y = np.zeros((*np.shape(xi), np.size(element))), np.zeros((*np.shape(eta), np.size(element)))

            for i in range(np.size(element)):
                x[..., i], y[..., i] = self.mapping(xi, eta, element=i)
            return x, y

        assert element < self.num_elements, "Element number out of bounds"
        # the logical spacing of the elements
        delta_x = 2.0 / self.n_x
        delta_y = 2.0 / self.n_y

        # x and y indices of the element
        index_x = np.ceil((element + 1) / self.n_y) - 1
        index_y = element % self.n_y

        x_left = -1 + delta_x * index_x
        y_left = -1 + delta_y * index_y

        x = (((xi + 1) * 0.5 * delta_x) + x_left + self.curvature * np.sin(np.pi * ((eta + 1) * 0.5 * delta_y + y_left)) *
             np.sin(np.pi * ((xi + 1) * 0.5 * delta_x + x_left)) + 1) * (self.x_bounds[1] - self.x_bounds[0]) * 0.5 + self.x_bounds[0]

        y = (((eta + 1) * 0.5 * delta_y) + y_left + self.curvature * np.sin(np.pi * ((eta + 1) * 0.5 * delta_y + y_left)) *
             np.sin(np.pi * ((xi + 1) * 0.5 * delta_x + x_left)) + 1) * (self.y_bounds[1] - self.y_bounds[0]) * 0.5 + self.y_bounds[0]

        print('xi, eta, x_left, y_left', xi, eta, x_left, y_left)
        return x, y

    # %% D_/D_
    def dx_dxi(self, xi, eta, element=None):
        if not isinstance(element, int):
            if element is None:
                element = np.arange(self.num_elements)
            dx_dxi_result = np.zeros((*np.shape(xi), np.size(element)))
            for i in range(np.size(element)):
                dx_dxi_result[..., i] = self.dx_dxi(xi, eta, element=i)
            return dx_dxi_result

        assert element <= self.num_elements, "Element number out of bounds"

        delta_x = 2.0 / self.n_x
        delta_y = 2.0 / self.n_y

        # x and y indices of the element
        index_x = np.ceil((element + 1) / self.n_y) - 1
        index_y = element % self.n_y

        x_left = -1 + delta_x * index_x
        y_left = -1 + delta_y * index_y

        dx_dxi_result = 0.5 * delta_x * \
            (self.x_bounds[1] - self.x_bounds[0]) * 0.5 + np.pi * delta_x * 0.5 * self.curvature * np.sin(np.pi * ((eta + 1) * 0.5 *
             delta_y + y_left)) * np.cos(np.pi * ((xi + 1) * 0.5 * delta_x + x_left)) * (self.x_bounds[1] - self.x_bounds[0]) * 0.5

        return dx_dxi_result

    # %% D_/D_
    def dx_deta(self, xi, eta, element=None):

        if not isinstance(element, int):
            if element is None:
                element = np.arange(self.num_elements)
            dx_deta_result = np.zeros((*np.shape(xi), np.size(element)))
            for i in range(np.size(element)):
                dx_deta_result[..., i] = self.dx_deta(xi, eta, element=i)
            return dx_deta_result

        assert element <= self.num_elements, "Element number out of bounds"

        delta_x = 2.0 / self.n_x
        delta_y = 2.0 / self.n_y

        # x and y indices of the element
        index_x = np.ceil((element + 1) / self.n_y) - 1
        index_y = element % self.n_y

        x_left = -1 + delta_x * index_x
        y_left = -1 + delta_y * index_y

        dx_deta_result = np.pi * delta_y * 0.5 * self.curvature * \
            np.cos(np.pi * ((eta + 1) * 0.5 * delta_y + y_left)) * \
            np.sin(np.pi * ((xi + 1) * 0.5 * delta_x + x_left)) * \
            (self.x_bounds[1] - self.x_bounds[0]) * 0.5

        return dx_deta_result

    # %% D_/D_
    def dy_dxi(self, xi, eta, element=None):
        if not isinstance(element, int):

            if element is None:
                element = np.arange(self.num_elements)

            dy_dxi_result = np.zeros((*np.shape(xi), np.size(element)))
            for i in range(np.size(element)):
                dy_dxi_result[..., i] = self.dy_dxi(xi, eta, element=i)
            return dy_dxi_result

        assert element <= self.num_elements, "Element number out of bounds"

        delta_x = 2.0 / self.n_x
        delta_y = 2.0 / self.n_y

        # x and y indices of the element
        index_x = np.ceil((element + 1) / self.n_y) - 1
        index_y = element % self.n_y

        x_left = -1 + delta_x * index_x
        y_left = -1 + delta_y * index_y

        dy_dxi_result = np.pi * delta_x * 0.5 * self.curvature * \
            np.sin(np.pi * ((eta + 1) * 0.5 * delta_y + y_left)) * \
            np.cos(np.pi * ((xi + 1) * 0.5 * delta_x + x_left)) * \
            (self.y_bounds[1] - self.y_bounds[0]) * 0.5

        return dy_dxi_result

    # %% D_/D_
    def dy_deta(self, xi, eta, element=None):
        if not isinstance(element, int):
            if element is None:
                element = np.arange(self.num_elements)
            dy_deta_result = np.zeros((*np.shape(xi), np.size(element)))
            for i in range(np.size(element)):
                dy_deta_result[..., i] = self.dy_deta(xi, eta, element=i)
            return dy_deta_result

        assert element <= self.num_elements, "Element number out of bounds"

        delta_x = 2.0 / self.n_x
        delta_y = 2.0 / self.n_y

        # x and y indices of the element
        index_x = np.ceil((element + 1) / self.n_y) - 1
        index_y = element % self.n_y

        x_left = -1 + delta_x * index_x
        y_left = -1 + delta_y * index_y

        dy_deta_result = 0.5 * delta_y * (self.y_bounds[1] - self.y_bounds[0]) * 0.5 + \
            np.pi * delta_y * 0.5 * self.curvature * \
            np.cos(np.pi * ((eta + 1) * 0.5 * delta_y + y_left)) * \
            np.sin(np.pi * ((xi + 1) * 0.5 * delta_x + x_left)) * \
            (self.y_bounds[1] - self.y_bounds[0]) * 0.5
        return dy_deta_result

# %% THE MAIN TEST PART
if __name__ == '__main__':
    mesh = CrazyMesh((5, 5), ((0, 1), (0, 1)), curvature=0.1)
    xi = np.linspace(-1, 1, 10)
    xi, eta = np.meshgrid(xi, xi)
    x, y = mesh.mapping(xi, eta, 0)
    # mesh.plot_mesh()
