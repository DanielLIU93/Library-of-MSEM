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
    def __init__(self, elements_layout, bounds_domain, curvature=0., info=None, name=None, pre_No=None, element_spacing=None):
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

        self._element_spacing = element_spacing

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

    @property
    def element_spacing(self):
        return self._element_spacing

    # Element layout
    def element_arrangement(self, n_x, n_y, element_nbr):
        # the logical spacing of the elements
        # deltax = 2.0 / n_x
        # deltay = 2.0 / n_y

        element_spacing = self.element_spacing

        # in the case when element_spacing is none, we will assign a simple uniform mesh
        if self.element_spacing is None:
            element_spacing = [(2.0/n_x)*np.ones((n_x,)), (2.0/n_y)*np.ones((n_y,))]

        assert [np.size(element_spacing[0]), np.size(element_spacing[1])] == [n_x, n_y], \
            'Input of element_spacing does not agree with the element_layout'

        # x and y indices of the element
        indexx = int(np.ceil((element_nbr + 1) / self.n_y) - 1)
        indexy = int(element_nbr % self.n_y)

        deltax = element_spacing[0]
        deltay = element_spacing[1]

        x_left = -1 + np.sum(deltax[:indexx])
        y_left = -1 + np.sum(deltay[:indexy])

        return x_left, y_left, deltax, deltay, indexx, indexy


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

        x_left, y_left, delta_x, delta_y, index_x, index_y = self.element_arrangement(self.n_x, self.n_y, element)

        x = (((xi + 1) * 0.5 * delta_x[index_x]) + x_left + self.curvature * np.sin(np.pi * ((eta + 1) * 0.5 * delta_y[index_y] + y_left)) *
             np.sin(np.pi * ((xi + 1) * 0.5 * delta_x[index_x] + x_left)) + 1) * (self.x_bounds[1] - self.x_bounds[0]) * 0.5 + self.x_bounds[0]

        y = (((eta + 1) * 0.5 * delta_y[index_y]) + y_left + self.curvature * np.sin(np.pi * ((eta + 1) * 0.5 * delta_y[index_y] + y_left)) *
             np.sin(np.pi * ((xi + 1) * 0.5 * delta_x[index_x] + x_left)) + 1) * (self.y_bounds[1] - self.y_bounds[0]) * 0.5 + self.y_bounds[0]

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

        x_left, y_left, delta_x, delta_y, index_x, index_y = self.element_arrangement(self.n_x, self.n_y, element)

        dx_dxi_result = 0.5 * delta_x[index_x] * (self.x_bounds[1]-self.x_bounds[0])*0.5 + \
                        np.pi*delta_x[index_x] * \
                        0.5 * self.curvature * np.sin(np.pi*((eta+1) * 0.5 * delta_y[index_y] + y_left)) * \
                        np.cos(np.pi*((xi + 1)*0.5*delta_x[index_x] + x_left)) * (self.x_bounds[1] - self.x_bounds[0]) * 0.5

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

        x_left, y_left, delta_x, delta_y, index_x, index_y = self.element_arrangement(self.n_x, self.n_y, element)

        dx_deta_result = np.pi * delta_y[index_y] * 0.5 * self.curvature * \
            np.cos(np.pi * ((eta + 1) * 0.5 * delta_y[index_y] + y_left)) * \
            np.sin(np.pi * ((xi + 1) * 0.5 * delta_x[index_x] + x_left)) * \
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

        x_left, y_left, delta_x, delta_y, index_x, index_y = self.element_arrangement(self.n_x, self.n_y, element)

        dy_dxi_result = np.pi * delta_x[index_x] * 0.5 * self.curvature * \
            np.sin(np.pi * ((eta + 1) * 0.5 * delta_y[index_y] + y_left)) * \
            np.cos(np.pi * ((xi + 1) * 0.5 * delta_x[index_x] + x_left)) * \
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

        x_left, y_left, delta_x, delta_y, index_x, index_y = self.element_arrangement(self.n_x, self.n_y, element)

        dy_deta_result = 0.5 * delta_y[index_y] * (self.y_bounds[1] - self.y_bounds[0]) * 0.5 + \
            np.pi * delta_y[index_y] * 0.5 * self.curvature * \
            np.cos(np.pi * ((eta + 1) * 0.5 * delta_y[index_y] + y_left)) * \
            np.sin(np.pi * ((xi + 1) * 0.5 * delta_x[index_x] + x_left)) * \
            (self.y_bounds[1] - self.y_bounds[0]) * 0.5
        return dy_deta_result

# %% THE MAIN TEST PART
if __name__ == '__main__':
    import functionals

    m, n = 13, 13
    ele_pnt_x = functionals.lobatto_quad(m)[0]
    ele_pnt_y = functionals.lobatto_quad(n)[0]

    ele_spa_x = [ele_pnt_x[i + 1] - ele_pnt_x[i] for i in range(0, m)]
    ele_spa_y = [ele_pnt_y[i + 1] - ele_pnt_y[i] for i in range(0, n)]

    Element_Spacing = [ele_spa_x, ele_spa_y]

    mesh = CrazyMesh((m, n), ((0, 1), (0, 1)), curvature=0, element_spacing=Element_Spacing)
    # xi = np.linspace(-1, 1, 10)
    # xi, eta = np.meshgrid(xi, xi)
    # x, y = mesh.mapping(xi, eta)
    mesh.plot_mesh()
