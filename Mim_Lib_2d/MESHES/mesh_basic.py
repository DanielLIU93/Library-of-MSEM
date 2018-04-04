"""
Parent for all meshes

@author: Created by Yi Zhang (张仪) [2017]
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
from abc import ABC
from tqdm import tqdm
from regions import Regions
import numpy as np
import functionals

import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'bwr'
plt.rc('text', usetex=False)
font = {'fontname': 'Times New Roman',
        'color': 'k',
        'weight': 'normal',
        'size': 12}


# %% THE CRAZY MESH CLASS BODY
class BasicMesh(ABC):
    """A Basic for all meshes"""

    # %% __INIT__
    def __init__(self, dim, R, elements_layout, info, name, pre_No):
        assert dim == 2, ' <MESH> : this is a 2-d mesh, make sure dim=2, now it is {}'.format(dim)
        self._dim = dim
        self._info = info
        self._name = name
        self._pre_No = pre_No
        self._anti_corner_singularity = None

        assert isinstance(R, Regions), " <MESH> : please feed me a Regions, now is {}".format(R)
        self._R = R

        self._lowest_numbering_version = self.R.lowest_numbering_version

        if isinstance(elements_layout, int):
            self._elements_layout = (elements_layout, elements_layout)
        elif isinstance(elements_layout, float):
            self._elements_layout = (int(elements_layout), int(elements_layout))
        elif isinstance(elements_layout, tuple):
            assert np.shape(elements_layout) == (2,), " <MESH> : elements_layout tuple shape wrong, now is {}".format(
                np.shape(elements_layout))
            self._elements_layout = (int(elements_layout[0]), int(elements_layout[1]))
        else:
            raise Exception(' <MESH> : elements_layout:{} not acceptable'.format(elements_layout))

        self._n_x, self._n_y = self._elements_layout

        self._num_elements = self.R.num_regions * self.elements_layout[0] * self.elements_layout[1]
        self._num_elements_each_region = self.elements_layout[0] * self.elements_layout[1]

        self._elements_map = None

        self._elements_map_r = None
        self._elements_map_r_local = None
        self._elements_numbering = None

    # %% PROPERTIES
    @property
    def dim(self):
        return self._dim

    @property
    def lowest_numbering_version(self):
        return self._lowest_numbering_version

    @property
    def info(self):
        return self._info

    @property
    def name(self):
        return self._name

    @property
    def pre_No(self):
        return self._pre_No

    @property
    def mesh_type(self):
        return self.__class__.__name__

    # %% Error conditioning handling
    @property
    def anti_corner_singularity(self):
        if self._anti_corner_singularity is not None:
            return self._anti_corner_singularity

        # If _anti_corner_singularity is None, automatically get it from the regions
        if self.R.regions_boundary_extreme != []:
            self._anti_corner_singularity = True
        else:
            self._anti_corner_singularity = False
        return self._anti_corner_singularity

    # you can also set it here
    @anti_corner_singularity.setter
    def anti_corner_singularity(self, OnOff):
        assert isinstance(OnOff, bool)
        self._anti_corner_singularity = OnOff

    # %% DISPLAY REGIONS
    @property
    def R(self):
        return self._R

    def get_boundary_ID(self, name):
        return self.R.get_boundary_ID(name)

    def get_region_ID(self, name):
        return self.R.get_region_ID(name)

    # %% NX, NY
    @property
    def elements_layout(self):
        return self._elements_layout

    @property
    def n(self):
        return self._elements_layout

    @property
    def n_x(self):
        return self._n_x

    @property
    def n_y(self):
        return self._n_y

    # %% NUM_ELEMENTS
    @property
    def num_elements(self):
        return self._num_elements

    @property
    def num_elements_each_region(self):
        return self._num_elements_each_region

    @property
    def elements_map(self):
        """
        self.elements_map, for refined elements map, use elements_map_r
        """
        if self._elements_map is not None:
            return self._elements_map

        elements_map_r = self.elements_map_r
        _elements_map = ()

        for i, j in enumerate(elements_map_r):
            _elements_map_i = ()
            for m, n in enumerate(j):
                if isinstance(n, str) and ' ' in n:
                    _elements_map_i += (int(n.split(' ')[0]),)
                else:
                    _elements_map_i += (n,)

            _elements_map += (_elements_map_i,)
        self._elements_map = _elements_map

        return self._elements_map

    # %% LOCAL ELEMENTS' MAP
    # This method is not very necessary at all
    @property
    def elements_map_r_local(self):
        if self._elements_map_r_local is not None: return self._elements_map_r_local
        self._elements_map_r_local = ()
        if self.elements_layout == (1, 1):
            # no local element_map_needed, the neighber of each element live in another reigon
            self._elements_map_r_local = ((),)

        elif self.elements_layout[0] >= 2 and self.elements_layout[1] >= 2:
            for i in range(self.n_x):
                for j in range(self.n_y):
                    s = j + i * self.n_y
                    if i == 0 and j == 0:  # south_west corner element
                        _elements_map_local_s = ('S', 1, 'W', self.n_y)
                    elif i == 0 and (0 < j < self.n_y - 1):  # west elements
                        _elements_map_local_s = (s - 1, s + 1, 'W', s + self.n_y)
                    elif i == 0 and j == self.n_y - 1:  # north_west element
                        _elements_map_local_s = (s - 1, 'N', 'W', s + self.n_y)
                    elif (0 < i < self.n_x - 1) and j == 0:  # south elements
                        _elements_map_local_s = ('S', s + 1, s - self.n_y, s + self.n_y)
                    elif (0 < i < self.n_x - 1) and (0 < j < self.n_y - 1):  # middle elements
                        _elements_map_local_s = (s - 1, s + 1, s - self.n_y, s + self.n_y)
                    elif (0 < i < self.n_x - 1) and j == self.n_y - 1:  # north elements
                        _elements_map_local_s = (s - 1, 'N', s - self.n_y, s + self.n_y)
                    elif i == self.n_x - 1 and j == 0:  # south-east elements
                        _elements_map_local_s = ('S', s + 1, s - self.n_y, 'E')
                    elif i == self.n_x - 1 and (0 < j < self.n_y - 1):  # east elements
                        _elements_map_local_s = (s - 1, s + 1, s - self.n_y, 'E')
                    elif i == self.n_x - 1 and j == self.n_y - 1:  # north-east elements
                        _elements_map_local_s = (s - 1, 'N', s - self.n_y, 'E')
                    else:
                        raise Exception(' <MESH> : Never reach here!!!!!')
                    self._elements_map_r_local += (_elements_map_local_s,)
        else:
            raise Exception(" <MESH> : FTTB, only accept self.elements_layout:({}) is (1,1) or (>1,>1)"
                            .format(self.elements_layout))

        return self._elements_map_r_local

    # %% ELEMENTS_MAP
    @property
    def elements_map_r(self):
        """'r' means refined. this gives the elements_map which has more information than the self.elements_map"""
        if self._elements_map_r is not None:
            return self._elements_map_r

        self._elements_map_r = ()
        print(" <MESH> : generate elements_map......")
        if self.elements_layout == (1, 1):
            for k in range(self.R.num_regions):
                the_location = 'SNWE ' + str(k) + ' '  # O means only one
                _elements_map_s = (
                    self._fec(the_location + 'S', 0, 0),
                    self._fec(the_location + 'N', 0, 0),
                    self._fec(the_location + 'W', 0, 0),
                    self._fec(the_location + 'E', 0, 0))

                self._elements_map_r += (_elements_map_s,)

        elif self.elements_layout[0] >= 2 and self.elements_layout[1] == 1:
            for k in tqdm(range(self.R.num_regions)):
                bne = k * self.num_elements_each_region
                #                the_location = str(k) + ' '
                for i in range(self.n_x):
                    s = bne + i  # local @element
                    if i == 0:  # left element
                        the_location = 'SNW ' + str(k) + ' '
                        _elements_map_s = (
                        self._fec(the_location + 'S', i, 0), self._fec(the_location + 'N', i, 0), self._fec(the_location + 'W', i, 0),
                        s + 1)
                    elif 0 < i < self.n_x - 1:  # center element
                        the_location = 'SN ' + str(k) + ' '
                        _elements_map_s = (self._fec(the_location + 'S', i, 0), self._fec(the_location + 'N', i, 0), s - 1, s + 1)
                    elif i == self.n_x - 1:  # right element
                        the_location = 'SNE ' + str(k) + ' '
                        _elements_map_s = (self._fec(the_location + 'S', i, 0), self._fec(the_location + 'N', i, 0), s - 1,
                                           self._fec(the_location + 'E', i, 0))
                    else:
                        raise Exception(' <MESH> : Never reach here!!!!!')

                    self._elements_map_r += (_elements_map_s,)

        elif self.elements_layout[0] == 1 and self.elements_layout[1] >= 2:
            for k in tqdm(range(self.R.num_regions)):
                bne = k * self.num_elements_each_region
                for j in range(self.n_y):
                    s = bne + j  # local @element
                    if j == 0:  # bottom element
                        the_location = 'SWE ' + str(k) + ' '
                        _elements_map_s = (self._fec(the_location + 'S', 0, j), s + 1, self._fec(the_location + 'W', 0, j),
                                           self._fec(the_location + 'E', 0, j))
                    elif 0 < j < self.n_y - 1:  # Heart element
                        the_location = 'WE ' + str(k) + ' '
                        _elements_map_s = (s - 1, s + 1, self._fec(the_location + 'W', 0, j), self._fec(the_location + 'E', 0, j))
                    elif j == self.n_y - 1:  # top element
                        the_location = 'NWE ' + str(k) + ' '
                        _elements_map_s = (s - 1, self._fec(the_location + 'N', 0, j), self._fec(the_location + 'W', 0, j),
                                           self._fec(the_location + 'E', 0, j))
                    else:
                        raise Exception(' <MESH> : Never reach here!!!!!')

                    self._elements_map_r += (_elements_map_s,)

        elif self.elements_layout[0] >= 2 and self.elements_layout[1] >= 2:
            for k in tqdm(range(self.R.num_regions)):
                bne = k * self.num_elements_each_region
                for i in range(self.n_x):
                    for j in range(self.n_y):
                        s = bne + j + i * self.n_y  # #element
                        if i == 0 and j == 0:  # south_west corner element
                            the_location = 'SW ' + str(k) + ' '
                            _elements_map_s = (
                            self._fec(the_location + 'S', i, j), s + 1, self._fec(the_location + 'W', i, j), s + self.n_y)
                        elif i == 0 and (0 < j < self.n_y - 1):  # west elements
                            the_location = 'W ' + str(k) + ' '
                            _elements_map_s = (s - 1, s + 1, self._fec(the_location + 'W', i, j), s + self.n_y)
                        elif i == 0 and j == self.n_y - 1:  # north_west element
                            the_location = 'NW ' + str(k) + ' '
                            _elements_map_s = (
                                s - 1, self._fec(the_location + 'N', i, j), self._fec(the_location + 'W', i, j), s + self.n_y)
                        elif (0 < i < self.n_x - 1) and j == 0:  # south elements
                            the_location = 'S ' + str(k) + ' '
                            _elements_map_s = (self._fec(the_location + 'S', i, j), s + 1, s - self.n_y, s + self.n_y)
                        elif (0 < i < self.n_x - 1) and (0 < j < self.n_y - 1):  # middle elements
                            the_location = 'M ' + str(k) + ' '
                            _elements_map_s = (s - 1, s + 1, s - self.n_y, s + self.n_y)
                        elif (0 < i < self.n_x - 1) and j == self.n_y - 1:  # north elements
                            the_location = 'N ' + str(k) + ' '
                            _elements_map_s = (s - 1, self._fec(the_location + 'N', i, j), s - self.n_y, s + self.n_y)
                        elif i == self.n_x - 1 and j == 0:  # south-east elements
                            the_location = 'SE ' + str(k) + ' '
                            _elements_map_s = (
                            self._fec(the_location + 'S', i, j), s + 1, s - self.n_y, self._fec(the_location + 'E', i, j))
                        elif i == self.n_x - 1 and (0 < j < self.n_y - 1):  # east elements
                            the_location = 'E ' + str(k) + ' '
                            _elements_map_s = (s - 1, s + 1, s - self.n_y, self._fec(the_location + 'E', i, j))
                        elif i == self.n_x - 1 and j == self.n_y - 1:  # north-east elements
                            the_location = 'NE ' + str(k) + ' '
                            _elements_map_s = (
                            s - 1, self._fec(the_location + 'N', i, j), s - self.n_y, self._fec(the_location + 'E', i, j))
                        else:
                            raise Exception(' <MESH> : Never reach here!!!!!')

                        self._elements_map_r += (_elements_map_s,)
        else:
            raise Exception(" <MESH> : For the time being, only accept elements_layout = (1,1) or (>1,>1)")

        return self._elements_map_r

    # %% FIND THE CONNECTION
    #   _fetch_the_element_connections
    def _fec(self, the_location, i, j):
        """
        #SUMMARY:
        # INPUTS:
            the_location :: for example, 'XX 3 S', the element is on the south of region #3
            i, j :: the local num of the element
        """

        the_location = the_location.split(' ')
        region_No = int(the_location[1])
        the_dict = {'S': 0, 'N': 1, 'W': 2, 'E': 3}
        position_str = the_location[2]
        position = the_dict[the_location[2]]
        what_is_here = self.R.regions_connections[region_No][position]
        if isinstance(what_is_here, str):
            # here is a boundary there
            return what_is_here
        elif isinstance(what_is_here, int):
            # here is another region
            abs_what_is_here = np.abs(what_is_here)
            if '+ ' + str(region_No) + ' ' + position_str in self.R.regions_connections_detials[abs_what_is_here]:
                index = self.R.regions_connections_detials[abs_what_is_here].index('+ ' + str(region_No) + ' ' + position_str)
                mark = '+'
            elif '- ' + str(region_No) + ' ' + position_str in self.R.regions_connections_detials[abs_what_is_here]:
                index = self.R.regions_connections_detials[abs_what_is_here].index('- ' + str(region_No) + ' ' + position_str)
                mark = '-'

            # abs_what_is_here: the #region
            # index: the position of the #region
            # mark: +, same orientation, -, opposite

            base_num_element = abs_what_is_here * self.num_elements_each_region
            if index == 0 and position in (0, 1):  # on the south of target region
                if mark == '+':
                    return '+' + str(self.n_y * i + base_num_element) + ' S'
                else:
                    return str(-1 * (self.n_y * (self.n_x - 1 - i) + base_num_element)) + ' S'
            elif index == 1 and position in (0, 1):  # on the north of target region
                if mark == '+':
                    return '+' + str(self.n_y * (i + 1) - 1 + base_num_element) + ' N'
                else:
                    return str(-1 * (self.n_y * (self.n_x - i) - 1 + base_num_element)) + ' N'
            elif index == 2 and position in (2, 3):  # on the west of target region
                if mark == '+':
                    return '+' + str(j + base_num_element) + ' W'
                else:
                    return str(-1 * (self.n_y - 1 - j + base_num_element)) + ' W'
            elif index == 3 and position in (2, 3):  # on the east of target region
                if mark == '+':
                    return '+' + str(self.num_elements_each_region - self.n_y + j + base_num_element) + ' E'
                else:
                    return str(-1 * (self.num_elements_each_region - j - 1 + base_num_element)) + ' E'
            # -------------------------------------------------------------------------------
            elif index == 0 and position in (2, 3):  # on the south of target region
                if mark == '+':
                    return '+' + str(self.n_y * j + base_num_element) + ' S'
                else:
                    return str(-1 * (self.n_y * (self.n_x - 1 - j) + base_num_element)) + ' S'
            elif index == 1 and position in (2, 3):  # on the north of target region
                if mark == '+':
                    return '+' + str(self.n_y * (j + 1) - 1 + base_num_element) + ' N'
                else:
                    return str(-1 * (self.n_y * (self.n_x - j) - 1 + base_num_element)) + ' N'
            elif index == 2 and position in (0, 1):  # on the west of target region
                if mark == '+':
                    return '+' + str(i + base_num_element) + ' W'
                else:
                    return str(-1 * (self.n_y - 1 - i + base_num_element)) + ' W'
            elif index == 3 and position in (0, 1):  # on the east of target region
                if mark == '+':
                    return '+' + str(self.num_elements_each_region - self.n_y + i + base_num_element) + ' E'
                else:
                    return str(-1 * (self.num_elements_each_region - i - 1 + base_num_element)) + ' E'

    # %% element numerbing gathering matrix
    @property
    def elements_numbering(self):
        """
        #SUMMARY: return the "gathering matrix" for elements
        #OUTPUTS:
            _elements_numbering :: 0-axis: global #element
                                   1-axis:
                                      first value: the region
                                      second value: local #element
        """
        if self._elements_numbering is not None: return self._elements_numbering
        self._elements_numbering = np.zeros(shape=(self.num_elements, 2), dtype=np.int64)
        for element in range(self.num_elements):
            region = int(np.floor(element / self.num_elements_each_region))
            local_element = element - region * (self.n_x * self.n_y)
            self._elements_numbering[element, 0] = region
            self._elements_numbering[element, 1] = local_element
        return self._elements_numbering

    # %% THE MAPING
    def mapping(self, xi, eta, element=None):
        pass

    # %% D_/D_
    def dx_dxi(self, xi, eta, element=None):
        pass

    # %% D_/D_
    def dx_deta(self, xi, eta, element=None):
        pass

    # %% D_/D_
    def dy_dxi(self, xi, eta, element=None):
        pass

    # %% D_/D_
    def dy_deta(self, xi, eta, element=None):
        pass

    # %% METRIC PART
    def g(self, xi, eta, element=None):
        metric_term = self.dx_dxi(xi, eta, element) * self.dy_deta(xi, eta, element) - \
                      self.dx_deta(xi, eta, element) * self.dy_dxi(xi, eta, element)
        return np.abs(metric_term)

    def g_11(self, xi, eta, element=None):
        g_11_result = (self.dx_deta(xi, eta, element) * self.dx_deta(xi, eta, element) +
                       self.dy_deta(xi, eta, element) * self.dy_deta(xi, eta, element)) / self.g(xi, eta, element) ** 2
        return g_11_result

    def g_12(self, xi, eta, element=None):
        g_12_result = -(self.dy_deta(xi, eta, element) * self.dy_dxi(xi, eta, element) +
                        self.dx_dxi(xi, eta, element) * self.dx_deta(xi, eta, element)) / self.g(xi, eta, element) ** 2
        return g_12_result

    def g_22(self, xi, eta, element=None):
        g_22_result = (self.dx_dxi(xi, eta, element) * self.dx_dxi(xi, eta, element) +
                       self.dy_dxi(xi, eta, element) * self.dy_dxi(xi, eta, element)) / self.g(xi, eta, element) ** 2
        return g_22_result

    # %% METRIC TENSOR PART
    def metric_tensor(self, xi, eta, element=None):
        """Calculate all the components of the metric tensor."""
        dx_deta = self.dx_deta(xi, eta, element)
        dx_dxi = self.dx_dxi(xi, eta, element)
        dy_deta = self.dy_deta(xi, eta, element)
        dy_dxi = self.dy_dxi(xi, eta, element)
        g = (dx_dxi * dy_deta -
             dx_deta * dy_dxi)
        g_11 = (dx_deta * dx_deta +
                dy_deta * dy_deta) / g
        g_12 = -(dy_deta * dy_dxi +
                 dx_dxi * dx_deta) / g
        g_22 = (dx_dxi * dx_dxi +
                dy_dxi * dy_dxi) / g
        """IMPORTENT: g_11, g_12, g_22 actually are g_11*g, g_12*g, g_22*g"""
        return g_11, g_12, g_22

    # %% METRIC TENSOR PART CORRECT VERSION
    def metric_tensor_correct(self, xi, eta, element=None):
        """Calculate all the components of the metric tensor."""
        dx_deta = self.dx_deta(xi, eta, element)
        dx_dxi = self.dx_dxi(xi, eta, element)
        dy_deta = self.dy_deta(xi, eta, element)
        dy_dxi = self.dy_dxi(xi, eta, element)
        g = (dx_dxi * dy_deta -
             dx_deta * dy_dxi)
        g_11 = (dx_deta * dx_deta +
                dy_deta * dy_deta) / g ** 2
        g_12 = -(dy_deta * dy_dxi +
                 dx_dxi * dx_deta) / g ** 2
        g_22 = (dx_dxi * dx_dxi +
                dy_dxi * dy_dxi) / g ** 2
        return g_11, g_12, g_22

    # %% JACOBIAN PART 
    def J11(self, xi, eta, element=None):
        return self.dx_dxi(xi, eta, element)

    def J12(self, xi, eta, element=None):
        return self.dx_deta(xi, eta, element)

    def J21(self, xi, eta, element=None):
        return self.dy_dxi(xi, eta, element)

    def J22(self, xi, eta, element=None):
        return self.dy_deta(xi, eta, element)

    def Jacobian(self, xi, eta, element=None):
        return (self.dx_dxi(xi, eta, element), self.dx_deta(xi, eta, element),
                self.dy_dxi(xi, eta, element), self.dy_deta(xi, eta, element))

    def detJ(self, xi, eta, element=None):
        return self.g(xi, eta, element)

    # %% invJACOBIAN PART 
    def invJ11(self, xi, eta, element=None):
        return (1 / self.g(xi, eta, element)) * self.dy_deta(xi, eta, element)

    def invJ12(self, xi, eta, element=None):
        return -(1 / self.g(xi, eta, element)) * self.dx_deta(xi, eta, element)

    def invJ21(self, xi, eta, element=None):
        return -(1 / self.g(xi, eta, element)) * self.dy_dxi(xi, eta, element)

    def invJ22(self, xi, eta, element=None):
        return (1 / self.g(xi, eta, element)) * self.dx_dxi(xi, eta, element)

    def invJacobian(self, xi, eta, element=None):
        g = self.g(xi, eta, element)
        return ((1 / g) * self.dy_deta(xi, eta, element),
                -(1 / g) * self.dx_deta(xi, eta, element),
                -(1 / g) * self.dy_dxi(xi, eta, element),
                (1 / g) * self.dx_dxi(xi, eta, element))

    def detinvJ(self, xi, eta, element=None):
        return 1 / self.g(xi, eta, element)

    # %% PLOT ITSELF
    def plot_mesh(self, regions=None, elements=None, plot_density=10,
                  internal_mesh_type=('gauss', (3, 3)),
                  return_mesh_data=False, show_plot=True):
        """
        #SUMMARY: A method to plot the mesh itself
        # INPUTS:
            @OPTIONAL:
                elements :: (None) elements to be plot.
                plot_density :: (10) How refined the plot will be
                internal_mesh_type :: (None) the meshtype plotted with in each element
                    EXAMPLES: internal_mesh_type = ('gauss', 5); an internal gauss mesh of
                    order 5 will be plotted

        #OUTPUTS: NONE
        #  RENEW: NONE
        #  NOTES:
        """
        print(" <MESH> : (partially) plot the mesh:")
        if isinstance(elements, int): elements = [elements, ]
        if elements.__class__.__name__ == 'ndarray' and np.ndim(elements) == 1:
            elements = np.int64(elements).tolist()
        assert isinstance(elements, list) or elements is None, ' <MESH> : elements_2b_plot should be a list or None'
        assert isinstance(regions, int) or regions is None, ' <MESH> : regions_2b_plot should be a list or None'

        # %% WHAT TO PLOT?
        if regions is None:
            regions = np.array([int(i) for i in range(self.R.num_regions)])
            if elements is None: elements = range(self.num_elements)
        elif isinstance(regions, int):
            if elements is None:
                elements = range(self.num_elements_each_region * regions, self.num_elements_each_region * (regions + 1))
            elif isinstance(elements, list):
                elements = (np.array(elements) + self.num_elements_each_region * regions).tolist()
            else:
                raise Exception('#elements wrong')
            regions = np.array([regions])
        else:
            raise Exception('#regions wrong')

        # regions: the regions to be plotted
        # elements: the elements to be plotted

        # %% PREPARE DATA FOR INTERNAL MESH AND ELEMENTS' BOUNDARIES
        xi = np.hstack((
            np.linspace(-1, 1, plot_density).reshape((plot_density, 1)),
            np.linspace(-1, 1, plot_density).reshape((plot_density, 1)),
            -np.ones((plot_density, 1)), np.ones((plot_density, 1))))
        eta = np.hstack((
            -np.ones((plot_density, 1)), np.ones((plot_density, 1)),
            np.linspace(-1, 1, plot_density).reshape((plot_density, 1)),
            np.linspace(-1, 1, plot_density).reshape((plot_density, 1))))

        if internal_mesh_type is None:
            pass
        elif internal_mesh_type[0] == 'gauss':
            nodes = getattr(functionals, 'gauss_quad')(internal_mesh_type[1][0])[0]
            nodes_grid_x = np.meshgrid(nodes, np.linspace(-1, 1, plot_density))
            nodes = getattr(functionals, 'gauss_quad')(internal_mesh_type[1][1])[0]
            nodes_grid_y = np.meshgrid(nodes, np.linspace(-1, 1, plot_density))
        elif internal_mesh_type[0] == 'lobatto':
            nodes = getattr(functionals, 'lobatto_quad')(internal_mesh_type[1][0])[0]
            nodes_grid_x = np.meshgrid(nodes, np.linspace(-1, 1, plot_density))
            nodes = getattr(functionals, 'lobatto_quad')(internal_mesh_type[1][1])[0]
            nodes_grid_y = np.meshgrid(nodes, np.linspace(-1, 1, plot_density))
        elif internal_mesh_type[0] == 'lobatto_separated':
            nodes = getattr(functionals, 'lobatto_quad')(internal_mesh_type[1][0])[0]
            nodes_grid_x = np.meshgrid(nodes, np.linspace(-1, 1, plot_density))
            nodes_grid_x = (nodes_grid_x[0] * 0.92, nodes_grid_x[1] * 0.92)
            nodes = getattr(functionals, 'lobatto_quad')(internal_mesh_type[1][1])[0]
            nodes_grid_y = np.meshgrid(nodes, np.linspace(-1, 1, plot_density))
            nodes_grid_y = (nodes_grid_y[0] * 0.92, nodes_grid_y[1] * 0.92)
        else:
            raise Exception(' <MESH> : internal_mesh_type:{} is wrong or not coded yet'.format(internal_mesh_type))

        if internal_mesh_type is not None:
            xi = np.hstack((xi, nodes_grid_x[0], nodes_grid_y[1]))
            eta = np.hstack((eta, nodes_grid_x[1], nodes_grid_y[0]))
        x, y = self.mapping(xi, eta, elements)

        # %% if show plot
        if show_plot is True:
            plt.figure()
            linewidth = 1.2
            for i in tqdm(range(np.size(elements))):
                if internal_mesh_type is not None:
                    plt.plot(x[:, 4:, i], y[:, 4:, i], 'r', linewidth=linewidth * 0.5)  # internal meshes

                plt.plot(x[:, :4, i], y[:, :4, i], 'k', linewidth=linewidth)  # ELEMENTS' EDGES

            # %% PLOT THE REGIONS' BOUNDARIES
            average_n = int((self._n_x + self._n_y) * 0.5)
            xi = np.hstack((
                np.linspace(-1, 1 + 2 * (self._n_x - 1), plot_density * average_n).reshape((plot_density * average_n, 1)),
                np.linspace(-1, 1 + 2 * (self._n_x - 1), plot_density * average_n).reshape((plot_density * average_n, 1)),
                -np.ones((plot_density * average_n, 1)), (1 + 2 * (self._n_x - 1)) * np.ones((plot_density * average_n, 1))))
            eta = np.hstack((
                -np.ones((plot_density * average_n, 1)), (1 + 2 * (self._n_y - 1)) * np.ones((plot_density * average_n, 1)),
                np.linspace(-1, 1 + 2 * (self._n_y - 1), plot_density * average_n).reshape((plot_density * average_n, 1)),
                np.linspace(-1, 1 + 2 * (self._n_y - 1), plot_density * average_n).reshape((plot_density * average_n, 1))))
            x_r, y_r = self.mapping(xi, eta, element=regions * self._num_elements_each_region)
            for i in range(np.size(regions)):
                plt.plot(x_r[:, :4, i], y_r[:, :4, i], 'b', linewidth=linewidth * 1.5)  # RETIONS' EDGES

            # %% EDIT THE PLOT
            plt.title(r"" + internal_mesh_type[0] + " mesh", fontdict=font)
            plt.xlabel(r'$x$');
            plt.ylabel(r'$y$');
            plt.axis('equal')
            plt.show()

        # %% not show plot, if return data?
        if return_mesh_data is True:
            return x, y

    # %% PLOT_METRIC
    def plot_Jacobian(self, xi=None, eta=None, plot_type='contourf', num_levels=20):
        """
        #SUMMARY: PLOT THE METRIC PART
        """
        if xi is None and eta is None:
            xi = eta = np.linspace(-1, 1, int(np.ceil(np.sqrt(10000 / self.num_elements)) + 1))
        xi, eta = np.meshgrid(xi, eta)
        x, y = self.mapping(xi, eta)
        dx_dxi_result = self.dx_dxi(xi, eta)
        dx_deta_result = self.dx_deta(xi, eta)
        dy_dxi_result = self.dy_dxi(xi, eta)
        dy_deta_result = self.dy_deta(xi, eta)

        plt.figure(figsize=(10, 8.8))
        # %% PLOT_METRIC dxdxi
        plt.subplot(221)
        levels = np.linspace(np.min(dx_dxi_result), np.max(dx_dxi_result) + 0.01, num_levels)
        for i in range(self.num_elements):
            getattr(plt, plot_type)(x[..., i], y[..., i], dx_dxi_result[..., i], levels=levels)
        plt.title(r"$\partial x/\partial \xi$", fontdict=font)
        plt.colorbar();
        plt.axis("equal");
        plt.show()
        # %% PLOT_METRIC dxdeta
        plt.subplot(222)
        levels = np.linspace(np.min(dx_deta_result), np.max(dx_deta_result) + 0.01, num_levels)
        for i in range(self.num_elements):
            getattr(plt, plot_type)(x[..., i], y[..., i], dx_deta_result[..., i], levels=levels)
        plt.title(r"$\partial x/\partial \eta$", fontdict=font)
        plt.colorbar();
        plt.axis("equal");
        plt.show()
        # %% PLOT_METRIC dydxi
        plt.subplot(223)
        levels = np.linspace(np.min(dy_dxi_result), np.max(dy_dxi_result) + 0.01, num_levels)
        for i in range(self.num_elements):
            getattr(plt, plot_type)(x[..., i], y[..., i], dy_dxi_result[..., i], levels=levels)
        plt.title(r"$\partial y/\partial \xi$", fontdict=font)
        plt.colorbar();
        plt.axis("equal");
        plt.show()
        # %% PLOT_METRIC dydeta
        plt.subplot(224)
        levels = np.linspace(np.min(dy_deta_result), np.max(dy_deta_result) + 0.01, num_levels)
        for i in range(self.num_elements):
            getattr(plt, plot_type)(x[..., i], y[..., i], dy_deta_result[..., i], levels=levels)
        plt.title(r"$\partial y/\partial \eta$", fontdict=font)
        plt.colorbar();
        plt.axis("equal");
        plt.show()

    # %% SELF_CHECKER
    @property
    def self_checker(self):
        """ SUMMARY: self-check at the end of __init__, including quality check """
        assert self.elements_layout[0] > 0 and self.elements_layout[1] > 0, \
            " <MESH> : elements_layout={} should be (>0, >0)".format(self.elements_layout)

    # %% MESH QULITY CHECK
    @property
    def quality(self):
        """ SUMMARY: for transfinite mapping mesh, mesh quality is equal to Regions quality """
        return self.R.quality
