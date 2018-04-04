# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 19:21:15 2017

@author: Yi Zhang, Created in 2017
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
"""
import numpy as np
import geometries
import warnings

# %% CLASS REGIONS MAP
class Regions(object):
    """
    #SUMMARY: A general mesh
    #PROPERTY:
        self.num_regions :: the number of regions we have

        self.regions :: tuple, self.regions[a][b][c]:
                        a -> #region
                        b -> ('SouthWest', 'SouthEast', 'NorthWest', 'NorthEast')
                        c -> [0] or [1], [0]-> x coordinate, [1]-> y coordinate

        self.regions_type :: tuple, self.regions_type[a][b][c]:
                             a -> #region
                             b -> ('South', 'North', 'West', 'East')
                             c -> [0]-> type name code:
                                            =-1: extreme situation, the edge is compacted into one point
                                            = 0: straight line
                                            = 1: anti_clock_wise arc
                                            = 2: clock_wise arc
                                  [1]-> type info
                                            straight line: empty
                                         uparc or downarc: a tuple: (p, q); arc center
                                         -1 extreme: a tuple, the coordinates of the point
        self.boundaries :: tuple
                            (  ((0,1), (1,3), ......), # BC1, containing North of #0 region, East of #3 region, ......
                               ((...), (...), ......), # BC2
                               ...... )

        self.boundaries_flat :: tuple: ((0,1), (1,3), ()....); all boundaries in a flat tuple

        self.regions_dof_map :: 2-d array:
                                1st dim -> regions
                                2rd dim -> 0,1,2,3 -> #num of SW, SE, NW, NE corner points

        self.regions_map :: tupel of shape (4, 4, 2), self.regions_map[a][b][c]
                            a -> #region
                            b -> ('South', 'North', 'West', 'East')
                            c -> a tuple: (#pt1, #pt2)
                                 pt1 is the left (or south) point of South, North (West, East) boundary of the region
        self.gamma :: tuple of shape (#regions, 2, 4)
                      2 -> means gamma and dgamma
                      4 -> means (d)gamma1, 2, 3, 4
    """
    def __init__(self, REGIONS=(None, None, None), regions_mapping=None):
        # regions, regions_type, boundaries = REGIONS
        self._dim = 2
        self._lowest_numbering_version = 1.0
        self._set_regions(REGIONS)
        self._boundaries_names_IDs = {}
        self._boundaries_IDs_names = {}
        
        self._num_boundaries_sections = None

        self.regions_mapping = regions_mapping

        self._generate_gamma()

        self._quality = None
        self._quality_angles = None
        self._regions_anlges = None

        self._connections_status = None

        self.self_checker

# PROPERTIES
    @property
    def dim(self):
        return self._dim

    @property
    def lowest_numbering_version(self):
        return self._lowest_numbering_version

# regions_mapping
    @property
    def regions_mapping(self):
        """
        regions_mapping will work when we generate MixedMesh
        the keys(regions names) in regions_mapping means these regions (not necessarily) have mappings other than Bilinear mapping
        For exmple: for a mesh of 5 regions, R0, R1, R2, R3, R4, R5, if the regions mapping is
             mesh.R.regions_mapping = {'R0': Bilinear, 'R1':Polar, 'R3':'Polar'}
             Then 'R0', 'R2', 'R4' are Bilinear mapping regions.
             'R1' and 'R3' are Polar mapping regions.
        """
        return self._regions_mapping

    @regions_mapping.setter
    def regions_mapping(self, regions_mapping):
        if regions_mapping is None:
            self._regions_mapping = {}
        else:
            assert isinstance(regions_mapping, dict), " <MESH> <REGION> : regions_mapping ({}) to be assigned must be a dict".format(regions_mapping.__class__.__name)
            mapping_keys = regions_mapping.keys()
            for i in mapping_keys:
                assert i in self.regions_names, " <MESH> <REGION> : regions_mapping: {}:{} is not a region name".format(i, regions_mapping[i])
                assert regions_mapping[i] in self.coded_regions_mapping_types, " <MESH> <REGION> : regions_mapping: {}:{} is not coded".format(i, regions_mapping[i])
            self._regions_mapping = regions_mapping

    @property
    def coded_regions_mapping_types(self):
        """
        reture the coded regions mapping, if we find a mapping not in this tuple,
        the program will immediately raise error! So when even coded a new mapping
        in MixedMesh, add the new mapping name in this tuple.
        """
        return ('Bilinear', 'Polar')

    # %% Regions
    @property
    def num_regions(self):
        return self._num_regions

    @property
    def regions(self):
        return self._regions

    @property
    def regions_type(self):
        return self._regions_type

    @property
    def regions_type_dict(self):
        return self._regions_type_dict

    @property
    def regions_names(self):
        return self._regions_names

    @property
    def regions_dict(self):
        return self._regions_dict

    @property
    def regions_type_explicit(self):
        return self._regions_type_explicit

    # %% BOUNDARIES RELATED
    @property
    def boundaries_names(self):
        return self._boundaries_names

    @property
    def boundaries_dict(self):
        return self._boundaries_dict

    @property
    def boundaries(self):
        return self._boundaries

    @property
    def boundaries_flat(self):
        return self._boundaries_flat

    @property
    def num_boundaries_each(self):
        return self._num_boundaries_each

    @property
    def num_boundaries_accumulated(self):
        return self._num_boundaries_accumulated

    @property
    def num_boundaries(self):
        return self._num_boundaries

    @property
    def num_boundaries_sections(self):
        """
        regions boundaries are divided into how many sections?
        """
        if self._num_boundaries_sections is None:
            self._num_boundaries_sections = np.shape(self.boundaries)[0]
        return self._num_boundaries_sections

    # %% REGIONS RELATED
    @property
    def num_regions_dof(self):
        return self._num_regions_dof

    @property
    def regions_flat(self):
        """
        #SUMMARY: self.regions_flat[i] returns the coordinate of regions_dof #i
        """
        return self._regions_flat

    @property
    def regions_dof_map(self):
        return self._regions_dof_map

    @property
    def regions_map(self):
        return self._regions_map

    @property
    def regions_connections(self):
        return self._regions_connections
    @property
    def regions_connections_detials(self):
        """
        Extended self.regions_connections, also including the S,N,W,E information
        """
        return self._regions_connections_details
    

    # %% GET REGION ID
    def get_region_ID(self, name):
        """
        #SUMMARY: give me the boundary name (see boundaries_dict), I return its ID (0,1,2,...)
        """
        try:
            return self._regions_names_list.index(name)
        except:
            raise Exception(" <MESH> <REGION> : {} is not in regions:{}".format(name, self.regions_names))

    # %% GET BOUNDARY ID
    def get_boundary_ID(self, name):
        """
        #SUMMARY: give me the boundary name (see boundaries_dict), I return its ID (0,1,2,...)
        """
        try:
            return self._boundaries_names_list.index(name)
        except:
            raise Exception(" <MESH> <REGION> : {} is not in boundaries:{}".format(name, self.boundaries_names))
    
    # %% GET REGION names
    def get_boundary_name(self, ID):
        """
        #SUMMARY: give me the boundary name (see boundaries_dict), I return its ID (0,1,2,...)
        """
        try:
            return self._boundaries_names[ID]
        except:
            raise Exception(" <MESH> <REGION> : {} is not not a valid boundary ID:{}".format(ID))

    # %% Boundaries Names & IDs list
    @property
    def boundaries_names_IDs(self):
        """get the dict corresponding to Boundary sections name and ID"""
        if self._boundaries_names_IDs != {}: return self._boundaries_names_IDs

        for i in range(self.num_boundaries_sections):
            self._boundaries_names_IDs[self.boundaries_names[i]] = i
        return self._boundaries_names_IDs
    
    # %% Boundaries IDs & names list
    @property
    def boundaries_IDs_names(self):
        """get the dict corresponding to Boundary sections name and ID"""
        if self._boundaries_IDs_names != {}: return self._boundaries_IDs_names

        for i in range(self.num_boundaries_sections):
            self._boundaries_IDs_names[str(i)] = self.boundaries_names[i]
        return self._boundaries_IDs_names
    
    # %% REGIONS SETTER
    def _set_regions(self, REGIONS):
        """
        #SUMMARY:store four corner points of each region
        # INPUTS: INPUTS = (regions, regions_type, boundaries)
                regions[a][b][c]: (of the same structure with self.regions)
                    a -> #region
                    b -> ('SouthWest', 'SouthEast', 'NorthWest', 'NorthEast')
                    c -> [0] or [1], [0]-> x coordinate, [1]-> y coordinate

                regions_type: a dictionary {'0N':(1, (2,3)), ......}
                        '0N':(1, (2,3)) means: #0 region, North boundary, of boundary type 1, with info (2,3)

                boundaries: (of the same structure with self.boundaries)
        #  RENEW:
                 self._regions = regions
                 self._regions_type, from regions_type
                 self.boundaries = boundaries
        """
        regions_dict, regions_type, boundaries_dict = REGIONS

        assert regions_dict.__class__.__name__ == 'dict', " <MESH> <REGION> : First entry of REGIONS should be the regions_dict, now is {}".format(regions_dict.__class__.__name__)
        self._regions_names =  tuple(regions_dict.keys())
        self._regions_names_list = list(regions_dict.keys())
        self._regions_dict = regions_dict
        regions = tuple(regions_dict.values())

        assert boundaries_dict.__class__.__name__ == 'dict', " <MESH> <REGION> : Third entry of REGIONS should be the boundaries_dict, now is {}".format(boundaries_dict.__class__.__name__)
        self._boundaries_names =  tuple(boundaries_dict.keys())
        self._boundaries_names_list = list(boundaries_dict.keys())
        self._boundaries_dict = boundaries_dict

        boundaries = tuple(boundaries_dict.values())
        boundaries_tuple = ()
        loca_dict = {'S':0, 'N':1, 'W':2, 'E':3}
        local_loca_store = []
        for i, b in enumerate(boundaries):
            boundaries_tuple_local = ()
            for j, c in enumerate(b):
                local_loca_store.append(c)
                assert 0 <= int(c[0]) < np.shape(regions)[0], " <MESH> <REGION> : Boundary_dict[{}][{}]: {} is out of regions range".format(self._boundaries_names_list[i], j, c)
                try:
                    boundaries_tuple_local += ((int(c[0]), loca_dict[c[1]]),)
                except KeyError:
                    raise KeyError(" <MESH> <REGION> : Boundary_dict[{}][{}]: {} is wrong".format(self._boundaries_names_list[i], j, c))
            boundaries_tuple += (boundaries_tuple_local,)
        assert len(local_loca_store) == len(set(local_loca_store)), " <MESH> <REGION> : At least one region boundary appears in two boundary sections"
        boundaries = boundaries_tuple

        # %% NO REGION
        if regions is None:
            raise Exception(' <MESH> <REGION> : Do not reach here!')
            self._num_regions = None
            self._regions = None
            self._regions_type = None
            self._regions_type_dict = None

            self._boundaries = None
            self._boundaries_flat = None
            self._num_boundaries_each = None
            self._num_boundaries_accumulated = None
            self._num_boundaries = None

            self._num_regions_dof = None
            self._regions_flat = None
            self._regions_dof_map = None

            self._regions_map = None

            self._regions_connections = None

        # %% HAVE A REGION
        else:
            assert np.shape(regions)[1]==4 and np.shape(regions)[2]==2, ' <MESH> <REGION> : regions size wrong'
            # %% num_regions
            self._num_regions = np.shape(regions)[0]

            # %% regions,  regions_flat and num_regions_dof
            self._regions = regions

            # %% region_type
            self._regions_type = ()
            self._regions_type_dict = {}
            for i in range(self._num_regions):
                South_type, North_type, West_type, East_type = (0,), (0,), (0,), (0,)
                if str(i)+'S' in regions_type:
                    South_type = regions_type[str(i)+'S']
                    self._regions_type_dict.update({str(i)+'S':regions_type[str(i)+'S']})
                if str(i)+'N' in regions_type:
                    North_type = regions_type[str(i)+'N']
                    self._regions_type_dict.update({str(i)+'N':regions_type[str(i)+'N']})
                if str(i)+'W' in regions_type:
                    West_type = regions_type[str(i)+'W']
                    self._regions_type_dict.update({str(i)+'W':regions_type[str(i)+'W']})
                if str(i)+'E' in regions_type:
                    East_type = regions_type[str(i)+'E']
                    self._regions_type_dict.update({str(i)+'E':regions_type[str(i)+'E']})

                if self.regions[i][0] == self.regions[i][1]:
                    South_type = (-1, self.regions[i][0])

                if self.regions[i][2] == self.regions[i][3]:
                    North_type = (-1, self.regions[i][3])

                if self.regions[i][0] == self.regions[i][2]:
                    West_type = (-1, self.regions[i][2])

                if self.regions[i][1] == self.regions[i][3]:
                    East_type = (-1, self.regions[i][1])

                self._regions_type += ((South_type,North_type,West_type,East_type),)

            if len(self._regions_type_dict) != len(regions_type):
                warnings.warn(" <MESH> <REGION> : At least one item in 'dict': regions_type is not valid")

            # %% boundaries and boundaries_flat
            if boundaries is ():
                self._boundaries = ()
                self._boundaries_flat = ()
            else:
                self._boundaries = boundaries
                self._boundaries_flat = ()
                self._num_boundaries_each = np.zeros(shape= np.shape(boundaries)[0], dtype=np.int64)
                self._num_boundaries_accumulated = np.zeros(shape= np.shape(boundaries)[0], dtype=np.int64)
                for i, j in enumerate(boundaries):
                    for k in j:
                        assert np.shape(k) == (2,), ' <MESH> <REGION> : boundaries data structure wrong, check if miss comma'
                        assert k[0] <= self._num_regions and k[1] <= 3, ' <MESH> <REGION> : (a,b) out of #num_regions range'
                    self._boundaries_flat += boundaries[i]

                    self._num_boundaries_each[i] = np.shape(boundaries[i])[0]
                    self._num_boundaries_accumulated[i] = np.shape(boundaries[i])[0]
                    if i > 0: self._num_boundaries_accumulated[i] += self._num_boundaries_accumulated[i-1]
                self._num_boundaries = np.max(self._num_boundaries_accumulated)

            # %% _num_regions_dof, _regions_flat and _regions_dof_map
            _regions_dof_map = np.zeros(shape=(self.num_regions, 4), dtype = np.int64)

            _regions_flat = ()
            k = 0
            for i, j in enumerate(regions):
                for position, coordinates in enumerate(j):
                    if coordinates not in _regions_flat:
                        _regions_dof_map[i, position] = k; k+=1
                        _regions_flat += (coordinates,)
                    else:
                        _regions_dof_map[i, position] = list(_regions_flat).index(coordinates)
            self._num_regions_dof = k
            self._regions_flat = _regions_flat
            self._regions_dof_map = _regions_dof_map

            # %% regions_map
            _regions_map = ()
            for i in range(self._num_regions):
                South = (self._regions_dof_map[i, 0], self._regions_dof_map[i, 1])
                North = (self._regions_dof_map[i, 2], self._regions_dof_map[i, 3])
                West = (self._regions_dof_map[i, 0], self._regions_dof_map[i, 2])
                East = (self._regions_dof_map[i, 1], self._regions_dof_map[i, 3])
                _regions_map += ((South, North, West, East),)
            self._regions_map = _regions_map

            # %% _regions_connections
            _regions_connections= ()
            _regions_connections_details = ()
            the_dict = {'0':'S', '1':'N', '2':'W', '3':'E'}
            for i in range(self._num_regions):
                _regions_connections_i = ()
                _regions_connections_details_i = [0,0,0,0]
                _regions_connections_local_check = ()
                for j in range(4):
                    if (i, j) in self._boundaries_flat:
                        for m, n in enumerate(self._boundaries):
                            if (i,j) in n:
                                _regions_connections_i += (str(m),)
                                _regions_connections_details_i[j]= self.get_boundary_name(m)
                    else:
                        for m, n in enumerate(self._regions_map):
                            if i == m: continue
                            if self._regions_map[i][j] in n:
                                _regions_connections_i += (m,)
                                _regions_connections_local_check += (m,)
                                SNWE = list(n).index(self._regions_map[i][j])
                                _regions_connections_details_i[j] = '+ '+str(m)+' '+the_dict[str(SNWE)]
                                
                            elif (self._regions_map[i][j][1], self._regions_map[i][j][0]) in n:
                                _regions_connections_i += (-m,)
                                _regions_connections_local_check += (m,)
                                SNWE = list(n).index((self._regions_map[i][j][1], self._regions_map[i][j][0]))
                                _regions_connections_details_i[j] = '- '+str(m)+' '+the_dict[str(SNWE)]

                if np.shape(_regions_connections_i) != (4,):
                    print(" <MESH> <REGION> : Something going wring in regions No.{}".format(i), _regions_connections_i)
                    raise Exception(' <MESH> <REGION> : Something is wrong in region No.{}'.format(i))
                
                if len(_regions_connections_local_check) != len(set(_regions_connections_local_check)):
                    print(" <MESH> <REGION> : R{}'s regions_connections\n\t{} is BAD, \n\t2 boundaries attached to the same region"
                          .format(i, _regions_connections_details_i))
                    self._lowest_numbering_version = 2.0
                
                _regions_connections += (_regions_connections_i,)
                _regions_connections_details += (_regions_connections_details_i,)

            self._regions_connections = _regions_connections
            self._regions_connections_details = _regions_connections_details

    # %% GAMMA GENERATOR
    def _generate_gamma(self):
        """
        #SUMMARY: A private method to generate the gamma
        # INPUTS:
        #OUTPUTS:
        #  RENEW:
            self.gamma
        """
        _gamma = ()
        _regions_type_explicit = ()
        self._regions_boundary_extreme = []
        # %%
        for i in range(self._num_regions):
            gamma_i = ()
            dgamma_i = ()
            _regions_type_explicit_i = ()
            for j in [0,1,2,3]:
                mapping_type = self._regions_type[i][j][0]
                if mapping_type == -1: # extreme
                    valu_x = self._regions_type[i][j][1][0]
                    valu_y = self._regions_type[i][j][1][1]
                    gamma = lambda o: (valu_x + 0*o, valu_y + 0*o)
                    dgamma = lambda o: (0*o, 0*o)
                    _regions_type_explicit_i += ('extreme',)
                    self._regions_boundary_extreme.append('R'+str(i)+'B'+str(j))
                    assert (i,j) in self.boundaries_flat, " <MESH> <REGION> : extreme boundaries must be on the boundary, otherwise the topology is confusing"

                elif mapping_type == 0: # straight line
                    gamma, dgamma = getattr(geometries, 'gamma_straightline')(
                            self._regions_flat[self._regions_map[i][j][0]],
                            self._regions_flat[self._regions_map[i][j][1]])
                    _regions_type_explicit_i += ('straightline',)
                elif mapping_type == 1: # anti_clock_wise arc
                    gamma, dgamma = getattr(geometries, 'gamma_arc_Anti_ClockWise')(
                            self._regions_type[i][j][1],
                            self._regions_flat[self._regions_map[i][j][0]],
                            self._regions_flat[self._regions_map[i][j][1]])
                    _regions_type_explicit_i += ('arc_Anti_ClockWise', )
                elif mapping_type == 2: # clock_wise arc
                    gamma, dgamma = getattr(geometries, 'gamma_arc_ClockWise')(
                            self._regions_type[i][j][1],
                            self._regions_flat[self._regions_map[i][j][0]],
                            self._regions_flat[self._regions_map[i][j][1]])
                    _regions_type_explicit_i += ('arc_ClockWise', )
                else:
                    raise Exception(' <MESH> <REGION> : Not applied yet')
                gamma_i += (gamma,)
                dgamma_i += (dgamma,)
            # %%
            _regions_type_explicit += (_regions_type_explicit_i,)

            _gamma += ((gamma_i, dgamma_i),)
        # %%
        self._gamma = _gamma
        self._regions_type_explicit = _regions_type_explicit

    # %% GET GAMMA
    @property
    def gamma(self):
        return self._gamma

    # %% Error_conditioning
    @property
    def regions_boundary_extreme(self):
        """
        store the extreme region edges:
        For example, R0B1, region #0, boundary [1], the norther boundary
        """
        return self._regions_boundary_extreme

    # %% SELF_CHECKER
    @property
    def self_checker(self):
        """
        #SUMMARY: self-checker. If OK, return nothing. If not OK, break the running.
                  If just fine, show the warngings.
        """
        assert np.shape(self.regions_type)[0] == np.shape(self.regions_map)[0] \
                == np.shape(self.regions)[0] == self.num_regions, \
                  " <MESH> <REGION> : some data structure wrong"

        if self.quality < 0.5:
            print(" <MESH> <REGION> : Warning, quality(=%.3f) too low" % self.quality)

        self.connections_status # %% To shown warnings of regions connections status

    # %% TOPOLOGY CHECKER
    @property
    def connections_status(self):
        """
        #SUMMARY: check if S,N,W,E are correctly attached, results are stored in
                  self._connections_status. If 0, bad; 0.5, warning; 1, good; -1, Boundary
        """
        if self._connections_status is not None: return self._connections_status

        self._connections_status = np.ones(shape=(self.num_regions, 4))
        for region_No in range(self.num_regions):
            for position in [0, 1, 2, 3]:
                what_is_here = self.regions_connections[region_No][position]
                if isinstance(what_is_here, str):
                    # here is a boundary there
                    self._connections_status[region_No, position] = -1
                    
                elif isinstance(what_is_here, int):
                    # here is another region
                    abs_what_is_here = np.abs(what_is_here)
                    if region_No in self.regions_connections[abs_what_is_here]:
                        index = list(self.regions_connections[abs_what_is_here]).index(region_No)
                        
                    elif -1 * region_No in self.regions_connections[abs_what_is_here]:
                        index = list(self.regions_connections[abs_what_is_here]).index(-1*region_No)
                        
                    else:
                        raise Exception(" <MESH> <REGION> : Regions.regions_connections is not correct, {} or {} is not in regions #.{}"
                                        .format(region_No, -1*region_No, abs_what_is_here))
                        
                    # abs_what_is_here: the #region of target region
                    # index: the position of the #region of target region

                    warning_dict = {'0':'S', '1':'N', '2':'W', '3':'E'}
                    
                    if position == 0 and index != 1:
                        if index == 0:
                            print(" <MESH> <REGION> : Warning, S of R#{} attach {} of R#{}"
                                  .format(region_No, warning_dict[str(index)], abs_what_is_here))
                            self._connections_status[region_No, position] = 0.5
                        else:
                            print(" <MESH> <REGION> : BAD!!!!, S of R#{} attach {} of R#{}"
                                  .format(region_No, warning_dict[str(index)], abs_what_is_here))
                            self._connections_status[region_No, position] = 0
                        
                    elif position == 1 and index != 0:
                        if index == 1:
                            print(" <MESH> <REGION> : Warning, N of R#{} attach {} of R#{}"
                                  .format(region_No, warning_dict[str(index)], abs_what_is_here))
                            self._connections_status[region_No, position] = 0.5
                        else:
                            print(" <MESH> <REGION> : BAD!!!!, N of R#{} attach {} of R#{}"
                                  .format(region_No, warning_dict[str(index)], abs_what_is_here))
                            self._connections_status[region_No, position] = 0
                        
                    elif position == 2 and index != 3:
                        if index == 2:
                            print(" <MESH> <REGION> : Warning, W of R#{} attach {} of R#{}"
                                  .format(region_No, warning_dict[str(index)], abs_what_is_here))
                            self._connections_status[region_No, position] = 0.5
                        else:
                            print(" <MESH> <REGION> : BAD!!!!, W of R#{} attach {} of R#{}"
                                  .format(region_No, warning_dict[str(index)], abs_what_is_here))
                            self._connections_status[region_No, position] = 0
                        
                    elif position == 3 and index != 2:
                        if index == 3:
                            print(" <MESH> <REGION> : Warning, E of R#{} attach {} of R#{}"
                                  .format(region_No, warning_dict[str(index)], abs_what_is_here))
                            self._connections_status[region_No, position] = 0.5
                            
                        else:
                            print(" <MESH> <REGION> : BAD!!!!, E of R#{} attach {} of R#{}"
                                  .format(region_No, warning_dict[str(index)], abs_what_is_here))
                            self._connections_status[region_No, position] = 0
                                
                    else:
                        pass
                    
        return self._connections_status

    # %% OVER ALL QUALITY
    @property
    def quality(self):
        """
        #SUMMARY: the over all factor of Regions quality.
        """
        if self._quality is not None: return self._quality

        self._quality = self.quality_angles
        return self._quality

    # %% REGION QUALITY in angles
    @property
    def quality_angles(self):
        """
        #SUMMARY: the Regions quality from the viewpoint of regions' four angles
        """
        if self._quality_angles is not None: return self._quality_angles
        # Obviously, the following algorithm for computing quality is very good.
        # Waiting for further update
        up_diff = np.abs(np.max(self.regions_angles)-90)
        do_diff = np.abs(np.min(self.regions_angles)-90)
        self._quality_angles = 1-np.max([up_diff, do_diff])/90
        return self._quality_angles

    @property
    def regions_angles(self):
        """
        #SUMMARY: compute the four corner angles of a region
        """
        if self._regions_anlges is not None: return self._regions_anlges

        self._regions_anlges = np.zeros(shape=(self.num_regions,4))
        for i, region in enumerate(self.regions):
            angleSW = geometries.anti_Clockwise_angle_between_two_lines(region[0], region[1], region[2])*180/np.pi
            angleSE = geometries.anti_Clockwise_angle_between_two_lines(region[1], region[3], region[0])*180/np.pi
            angleNW = geometries.anti_Clockwise_angle_between_two_lines(region[2], region[0], region[3])*180/np.pi
            angleNE = geometries.anti_Clockwise_angle_between_two_lines(region[3], region[2], region[1])*180/np.pi
            self._regions_anlges[i,:]=[angleSW, angleSE, angleNW, angleNE]
        return self._regions_anlges