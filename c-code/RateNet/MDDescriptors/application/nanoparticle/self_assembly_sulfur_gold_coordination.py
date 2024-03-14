# -*- coding: utf-8 -*-
"""
self_assembly_sulfur_gold_coordination.py
The purpose of this script is to analyze the sulfur gold coordination on self-assembly structure.

Written by: Alex K. Chew (alexkchew@gmail.com, 08/20/2018)

"""

### IMPORTING MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import numpy as np
import MDDescriptors.core.calc_tools as calc_tools # calc tools
import MDDescriptors.core.read_write_tools as read_write_tools # Reading itp file tool
import mdtraj as md
### IMPORTING NANOPARTICLE STRUCTURE CLASS
from MDDescriptors.application.nanoparticle.nanoparticle_structure import nanoparticle_structure
### IMPORTING GLOBAL VARIABLES
from MDDescriptors.application.nanoparticle.global_vars import GOLD_SULFUR_CUTOFF, GOLD_GOLD_CUTOFF_BASED_ON_SHAPE, NP_WORKING_DIR
## IMPORTING MULTI TRAJ FUNCTIONS
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles, load_multi_traj_multi_analysis_pickle, load_pickle_for_analysis
## BUNDLING FUNCTIONS
from MDDescriptors.application.nanoparticle.nanoparticle_find_bundled_groups import cluster_DBSCAN, find_group_assignments
## IMPORTTING TOOLS FROM SULFUR GOLD COORDINATION
from MDDescriptors.application.nanoparticle.nanoparticle_sulfur_gold_coordination import find_nearest_gold_atoms_to_sulfur

### FUNCTION TO PLOT THE CONVEX HULL
def plot_convex_hull_color(  hull, color = 'k', ax = None, label = None, **line_styles ):
    '''
    The purpose of this function is to print the convex hull based on a color
    INPUTS:
        hull: [class]
            ConvexHull class 
        color: [str]
            color to color the hull
        ax: [ax]
            axis from matplotlib
        label: [str]
            label for the plot
        line_styles: [dict]
            line styles you want for your hull
    OUTPUTS:
        ax: [ax]
            updated axis
    '''
    if ax == None:
        from MDDescriptors.core.plot_tools import create_3d_axis_plot_labels
        fig, ax = create_3d_axis_plot_labels( labels = ['x (nm)', 'y (nm)', 'z (nm)'] )
    ## DEFINING POINTS
    points = hull.points
    ## LOOPING THROUGH EACH OF THE SIMPLICES
    for idx, simplex in enumerate(hull.simplices):
        if idx != 0:
            label = None
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], color = color, label = label, **line_styles)
    
    return ax

### FUNCTION TO GET CMAP
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.
    This function is useful to generate colors between red and purple without having to specify the specific colors
    USAGE:
        ## GENERATE CMAP
        cmap = get_cmap(  len(self_assembly_coord.gold_facet_groups) )
        ## SPECIFYING THE COLOR WITHIN A FOR LOOP
        for ...
            current_group_color = cmap(idx) # colors[idx]
            run plotting functions
    '''
    ## IMPORTING FUNCTIONS
    import matplotlib.pyplot as plt
    return plt.cm.get_cmap(name, n + 1)

### FUNCTION TO FIND FACETS TO EDGES RATIO
def calc_facets_to_edges_ratio(traj, gold_sulfur_atom_pairs, gold_atom_index, sulfur_atom_index, gold_surface_facet_vs_edge_classification, 
                               maximum_num_gold_atoms = 3, periodic = True):
    '''
    The purpose of this function is to calculate the facets to edges ratio for each sulfur atom.
        - We will do this by finding nearby sulfur-gold using a cutoff (self.gold_sulfur_cutoff)
        - This step is expected to be the slow step (due to distance calculations)
    INPUTS:
        traj: trajectory from md.traj
        gold_sulfur_atom_pairs: [np.array, shape=(num_pairs,2)] atom pairs between gold and sulfur atoms
        gold_atom_index: [np.array, shape=(num_gold_atoms, 1)] gold atom index used to generate the atom pairs
        sulfur_atom_index: [np.array, shape=(num_sulfur_atoms, 1)] sulfur atom index used to generate the atom pairs
        periodic: [logical, default=True] True if you want PBCs to be accounted for
        maximum_num_gold_atoms: [int, default=3] number of maximum gold atoms to search for
    OUTPUTS:
        ratio_of_facet_surface_atoms: [np.array, shape=(num_frames, num_sulfur_atoms)] ratio of facet atoms to total surrounding atoms
            - can contain nan's. If this is the case, then the sulfur atom is not nearby any of the gold atoms based on your cutoff.
            - all numpy nan's will be ignored, so no error message will be displayed
        distances: [np.array, shape=(num_frames, num_gold_atoms, num_sulfur_atoms)]
            distance matrix for gold-sulfur
    
    '''
    ## SETTING ERRORS FOR DIVISION BY ZERO OFF
    import numpy as np
    np.seterr(divide='ignore', invalid='ignore')
    
    ## CALCULATING DISTANCES BETWEEN GOLD AND SULFUR ATOMS
    distances = md.compute_distances( traj = traj, atom_pairs = gold_sulfur_atom_pairs, periodic = periodic )
    
    ## RESHAPING THE DISTANCE MATRIX
    distances = distances.reshape( ( distances.shape[0], len(sulfur_atom_index), len(gold_atom_index)   )  )
    ## RETURNS: num_frames, num_sulfur_atoms, num_gold_atoms
    ## TRANSPOSING DISTANCE MATRIX TO CORRECTLY RUN ON GOLD ATOMS
    distances = np.transpose( distances, (0, 2, 1) )
    ## RETURNS: num_frames, num_gold_atoms, num_sulfur_atoms
    
    ## RATIO USING NEAREST GOLD ATOMS
    indices_of_nearest = find_nearest_gold_atoms_to_sulfur(distances, maximum_num_gold_atoms = maximum_num_gold_atoms)
    ## FINDING RATIO
    ratio_of_facet_surface_atoms = np.mean(gold_surface_facet_vs_edge_classification[indices_of_nearest], axis =1)
        
    return ratio_of_facet_surface_atoms, distances

### FUNCTION TO FIND INTERSECTION OF GROUPS TO ATOM INDICES
def find_intersect_group_list_to_atom_index(group_dict, atom_index_list):
    '''
    The purpose of this function is to take a group dictionary, and divide an atom index based on the list
    INPUTS:
        group_dict: [dict]
            dictionary containing the atom index as a form of a group (e.g. '1', etc.). Within each group, you should have a numpy array containing the atom indices
        atom_indices: [np.array]
            atom index that you want to divide into groups
    OUTPUTS:
        atom_index_dict: [dict]
            dictionary with the same keys as 'group_dict', containing the atom indices that are split based on intersections
    '''
    ## DEFINING NEW DICTIONARY
    atom_index_dict = dict()
    ## LOOPING THROUGH KEYS OF GROUP DICT
    for each_key in group_dict:
        ## FINDING ALL SIMILAR ARRAYS
        similar_values = np.isin(  atom_index_list, group_dict[each_key]).nonzero()
        ## STORING SIMILAR VALUES
        atom_index_dict[each_key] = similar_values
    return atom_index_dict

### FUNCTION TO CONVERT EACH KEY BASED ON A DICTIONARY
def convert_each_key_in_dict_given_positions(dict_to_convert , conversion_indices ):
    '''
    The purpose of this function is to convert a given list of keys with the corresponding indexes into the new set of indices
    This function is useful in the case where you have position indices and have atom indices
    INPUTS:
        dict_to_convert: [dict]
            dictionary to convert
        conversion_indices: [np.array]
            indices used to convert
    OUTPUTS:
        converted_dict: [dict]
            dictionary with the same keys as 'dict_to_convert'
    '''
    ## CREATING NEW DICTIONARY
    converted_dict=dict()
    ## LOOPING THROUGH EACH KEY
    for each_key in dict_to_convert:
        ## USING THE INDICES TO CREATE A NEW KEY
        converted_dict[each_key] = conversion_indices[dict_to_convert[each_key]]
        
    return converted_dict

### FUNCTION TO UPDATE THE EDGE ATOM AND FACET ATOM INDICES
def update_edge_atom_indices( traj, frame, gold_atom_index, gold_facet_atom_indices, gold_edge_atom_indices , gold_gold_cutoff ):
    '''
    The purpose of this function is to add one layer of facet gold atoms to edge gold atom
    INPUTS:
        traj: [class object]
            trajectory from md.traj
        frame: [int]
            frame of the trajectory that you are interested in studying
        gold_atom_index: [np.array, shape=(num_gold_atoms, 1)]
            gold atom index in reference to the trajectory
        gold_facet_atom_indices: [np.array, shape=(num_facet_gold_atoms, 1)]
            gold atom index of facets with respect to gold atom index
        gold_edge_atom_indices: [np.array, shape=(num_edge_gold_atoms, 1)]
            gold atom index of edge with respect to gold atom index
        gold_gold_cutoff: [float]
            gold-gold cutoff to be considered coordinated with each other
    OUTPUTS:
        gold_atom_facet_atom_indices_updated: [np.array, shape=(num_facet_gold_atoms, 1)]
            gold atom index of facet that has been updated
        gold_atom_edge_atom_indices_updated: [np.array, shape=(num_facet_gold_atoms, 1)]
            gold atom edge index
    '''
    ## CALCULATING DISTANCES
    gold_edge_facet_atom_distances = calc_tools.calc_pair_distances_between_two_atom_index_list( 
                                            traj            =  traj[frame],
                                            atom_1_index    = gold_atom_index[gold_edge_atom_indices],
                                            atom_2_index    = gold_atom_index[gold_facet_atom_indices],
                                            periodic        = True,                                            
                                            )[0] ## RETURNS NUM_EDGE VS. NUM_FACET

    ## FINDING ALL COLUMNS THAT ARE WITHIN A GOLD CUTOFF DISTANCE
    facet_gold_indices_within_edge_cutoff = gold_facet_atom_indices[np.unique(np.argwhere(gold_edge_facet_atom_distances <= gold_gold_cutoff)[:,1])]

    ## UPDATING FACET ATOM INDICES AND EDGE ATOM INDICES
    gold_atom_facet_atom_indices_updated = np.setdiff1d( gold_facet_atom_indices , facet_gold_indices_within_edge_cutoff )

    ## UPDATING EDGE ATOMS AND SORTING
    gold_atom_edge_atom_indices_updated = np.sort( np.append( gold_edge_atom_indices,facet_gold_indices_within_edge_cutoff  ) )
    
    return gold_atom_facet_atom_indices_updated, gold_atom_edge_atom_indices_updated


##########################################################
### CLASS FUNCTION TO ANALYZE SULFUR-GOLD COORDINATION ###
##########################################################
class self_assembly_sulfur_gold_coordination:
    '''
    The purpose of this function is to analyze the sulfur atoms on the gold core based on coordination numbers.
    INPUTS:
        traj_data: Data taken from import_traj class
        pickle_loading_file: [str]
            name of the pickle loading file (currently)
        analysis_classes: [list]
            list of list of analysis classes and corresponding dates to load the pickles
        gold_coordination_surface: [list]
            list of integers that denote the coordination numbers to be considered as a surface gold atom            
        frame: [int, default = -1]
            frame you are interested in studying the sulfur gold coordination --- used to speed up the calculation
        maximum_num_gold_atoms: [int]
            maximum nubmer of gold atoms to find the bonding between sulfur and gold atoms
        min_gold_samples: [int]
            minimum gold samples to be considered as part of a cluster
        gold_sulfur_cutoff: [float]
            gold-sulfur cutoff
        debug: [logical, default=False]
            True if you want to turn on debugging of this script, which will run through plots to ensure consistency
        want_expand_edges: [logical, default=False]
            True if you want to test expanding the edges by nearest neighbors
        save_disk_space: [logical, default=True]
            True if you want to save disk space, thus removing unnecessary variables
    OUTPUTS:
        ### STORING INPUTS
            self.maximum_num_gold_atoms, self.gold_sulfur_cutoff, self.frame, self.min_gold_samples
        ### EXTRACTION OF SELF ASSEMBLY STRUCTURE DATA
            self.gold_gold_cutoff: [float]
                gold-gold cutoff used for the coordination numbers
            self.gold_edge_atom_indices: [np.array, shape=(num_edge, 1)] 
                atom indices of gold edge atoms
            self.gold_facet_atom_indices: [np.array, shape=(num_facets, 1)] 
                atom indices of gold facet atoms
        
    
    ALGORITHM:
        - load self-assembly structure <-- has information about the coordination numbers of gold!
        - select the last trajectory to analyze
        - CALCULATING TOTAL SULFURS ON THE FACETSl
            - use gold coordination scripts to calculate the sulfurs that are coordinated to the gold atoms
        - CALCULATING TOTAL AREA OF THE GOLD FACETS
            - get all gold atoms that are coordinated with specific indices
            - make a plot of the gold atoms to ensure single planar surfaces
            - use gold-gold bond distances to classify the facets
            - get area of each of the facets
    '''
    ### INITIALIZING
    def __init__(self, traj_data, pickle_loading_file, analysis_classes, 
                 maximum_num_gold_atoms, gold_sulfur_cutoff, min_gold_samples,
                  frame = -1, debug = False, 
                  want_expand_edges = False,
                  save_disk_space = True):
        ## STORING INITIAL VARIABLES
        self.maximum_num_gold_atoms     = maximum_num_gold_atoms
        self.gold_sulfur_cutoff         = gold_sulfur_cutoff
        self.frame                      = frame
        self.min_gold_samples           = min_gold_samples
        
        ## DEFINING TRAJECTORY
        traj = traj_data.traj
        
        ## RELOADING STRUCTURE CLASS
        self.self_assembly_structure = load_pickle_for_analysis(
                                                                analysis_classes = analysis_classes, 
                                                                function_name = 'self_assembly_structure', 
                                                                pickle_file_name = pickle_loading_file,
                                                                conversion_type = None,
                                                                current_work_dir = NP_WORKING_DIR,
                                                                )
        ##########################
        ### EXTRACTION OF DATA ###
        ##########################
        
        ## DEFINING SULFUR INDEX
        self.sulfur_atom_index = self.self_assembly_structure.sulfur_index
        ## DEFINING GOLD INDEX
        self.gold_atom_index = self.self_assembly_structure.gold_optimize_index
        
        ## DEFINING THE CUTOFF
        self.gold_gold_cutoff = self.self_assembly_structure.gold_gold_coordination['gold_gold_cutoff']
        ## DEFINING ATOM INDICES OF ALL GOLD SURFACE INDICES
        self.gold_surface_atom_indices = self.self_assembly_structure.gold_gold_coordination['gold_surface_indices']
        ## DEFINING ATOM INDICES OF EDGES
        self.gold_edge_atom_indices = self.gold_surface_atom_indices[self.self_assembly_structure.gold_gold_surface_facet_edge_dict['gold_surface_edge_index']]
        ## DEFINING THE ATOM INDICES OF THE FACETS
        self.gold_facet_atom_indices = self.gold_surface_atom_indices[self.self_assembly_structure.gold_gold_surface_facet_edge_dict['gold_surface_facet_index']]
        
        ## UPDATING INDEX IF YOU WANT TO EXPAND EDGES
        if want_expand_edges == True:
            self.gold_facet_atom_indices, self.gold_edge_atom_indices = update_edge_atom_indices( 
                                                traj = traj,
                                                frame = self.frame,
                                                gold_atom_index = self.gold_atom_index,
                                                gold_facet_atom_indices = self.gold_facet_atom_indices,
                                                gold_edge_atom_indices = self.gold_edge_atom_indices,
                                                gold_gold_cutoff = self.gold_gold_cutoff,
                    )
        
        #############################
        ### COMPUTING GOLD FACETS ###
        #############################
        
        ## FINDING GOLD FACET DISTANCE MATRIX
        self.calc_gold_facet_distance_matrix(traj = traj)
        
        ## FINDING UNIQUE FACETS AROUND THE GOLD CORE self.gold_gold_cutoff 0.50
        self.gold_facet_labels = cluster_DBSCAN(X = self.gold_facet_distance_matrix, eps = self.gold_gold_cutoff , min_samples = self.min_gold_samples)
        
        ## FINDING GROUP ASSIGNMENTS
        self.gold_facet_groups = find_group_assignments( self.gold_facet_labels )
        
        ## REMOVING ANY UNLABELLED 
        if '-1' in self.gold_facet_groups.keys():
            ## REMOVAL OF -1 FROM FACET GROUP
            del self.gold_facet_groups['-1']
        
        ## FINDING COLORS FOR EACH OF THE GROUPS (Plotting purposes)
        self.cmap = get_cmap(  len(self.gold_facet_groups) )       
        
        #######################################
        ### FINDING POSITIONS FOR DEBUGGING ###
        #######################################
        
        ## FINDING POSITIONS OF THE GOLD ATOMS
        self.gold_facet_atom_positions =  traj.xyz[self.frame, self.gold_atom_index[self.gold_facet_atom_indices] , :]
        self.gold_edge_atom_positions  =  traj.xyz[self.frame, self.gold_atom_index[self.gold_edge_atom_indices] , :]
        
        ## PLOTTING THE ATOMS
        if debug == True:
            self.plot3d_surface_colored_facets()
        
        ### CALCULATING THE CONVEX HULL
        self.calc_gold_facet_convex_hull()
        
        ## FINDING CONVEX HULL OF ENTIRE SURFACE
        self.calc_gold_surface_convex_hull(traj = traj)
        
        ## PLOTTING THE CONVEX HULL
        if debug == True:
            self.plot3d_surface_colored_with_convex_hull()
        
        ###########################
        ### GOLD-SULFUR BONDING ###
        ###########################
        ## CALCULATING THE SULFUR GOLD DISTANCE MATRIX
        self.calc_sulfur_gold_bonding_distances(traj = traj)
        ## FINDING NEARBY SULFUR ATOM INDEX
        self.find_nearby_sulfur_indices()
        ## FINDING POSITIONS OF THE NEARBY SULFUR
        self.nearby_sulfur_coordinates = traj.xyz[self.frame, self.sulfur_atom_index_within_cutoff , :]
        
        ## PLOTTING THE SULFUR ON THE GOLD
        if debug == True:
            self.plot3d_sulfur_on_gold()
            
        ################################
        ### GOLD_SULFUR COORDINATION ###
        ################################
        ## DEFINING GOLD INDICES TO LOOK AT 
        # self.gold_sulfur_coord_gold_serial_indices =np.intersect1d(self.gold_atom_index, self.gold_surface_atom_indices) # INDICES OF SURFACE GOLD ATOMS
        ## FINDING ALL SULFUR WITHIN FACETS
        self.find_gold_sulfur_coordination_groups()
        ## PLOTTING
        if debug == True:
            self.plot3d_sulfur_per_facet_basis()
        
        #######################################
        ### RATIO OF FACET TO SURFACE ATOMS ###
        #######################################
        ## GETTING ATOM PAIRS BETWEEN GOLD AND SULFUR
        gold_sulfur_atom_pairs = calc_tools.create_atom_pairs_list( atom_1_index_list = self.gold_atom_index[self.gold_surface_atom_indices],
                                                                         atom_2_index_list = self.sulfur_atom_index_within_cutoff,)
        
        ## FINDING RATIO OF FACET TO SURFACE ATOMS
        self.ratio_of_facet_surface_atoms, self.surface_gold_sulfur_distances = calc_facets_to_edges_ratio(  traj = traj[self.frame],
                                                                                                             gold_sulfur_atom_pairs = gold_sulfur_atom_pairs,
                                                                                                             gold_atom_index = self.gold_atom_index[self.gold_surface_atom_indices],
                                                                                                             sulfur_atom_index = self.sulfur_atom_index_within_cutoff,
                                                                                                             gold_surface_facet_vs_edge_classification = self.self_assembly_structure.gold_gold_surface_facet_edge_dict['gold_surface_facet_vs_edge_classification'],
                                                                                                             maximum_num_gold_atoms = self.maximum_num_gold_atoms,
                                                                                                             ) # Using only one frame
        
        ## REMOVING THE TIME FRAMES
        self.ratio_of_facet_surface_atoms = self.ratio_of_facet_surface_atoms[0]
        
        ## PLOTTING RATIO OF FACET SURFACE ATOMS
#        if debug == True:
#            self.plot3d_sulfur_color_coded_on_gold()
        
        ####################################################
        ### COMPUTING DESITY OF GOLD SULFUR COORDINATION ###
        ####################################################
    
        ## FINDING TOTAL SULFURS AT EACH FACET
        self.results_num_sulfur_each_facet =np.array(
                                                    [ len(self.gold_sulfur_coord_split_into_facets_serial_index_dict[each_key][0]) for each_key in self.gold_sulfur_coord_split_into_facets_serial_index_dict]
                                                )
        ## FINDING TOTAL RATIO PER FACET
        self.results_planar_density_angs_per_lig_per_facet = self.gold_facet_convex_hull_surface_area_per_facet/ self.results_num_sulfur_each_facet *100
        
        ## FINDING TOTAL SULFURS AT THE FACET
        self.total_facet_sulfurs = np.sum(self.results_num_sulfur_each_facet) # (self.ratio_of_facet_surface_atoms==1).sum()
        ## FINDING TOTAL EDGE SULFURS
        self.total_edge_sulfurs = len(self.sulfur_atom_index_within_cutoff) -  self.total_facet_sulfurs  # (self.ratio_of_facet_surface_atoms==0).sum()
        
        ## COMPUTING PLANAR AREA DENSITY
        self.planar_density_angs_per_lig = self.gold_facet_convex_hull_surface_area / self.total_facet_sulfurs *100
        
        ## COMPUTING THE HULL SURFACE AREA
        self.gold_facet_convex_hull_edge_surface_area = self.self_assembly_structure.surface_area_hull - self.gold_facet_convex_hull_surface_area 
        self.planar_density_angs_per_lig_edge = self.gold_facet_convex_hull_edge_surface_area / self.total_edge_sulfurs * 100

        ## PRINTING SUMMARY
        self.print_summary()
        
        

        return
    
    ### FUNCTION TO PRINT SUMMARY
    def print_summary(self):
        ''' This function prints a summary '''
        print("---- SURFACE AREAS ----")
        print("Total surface area of the planes: %.2f nm^2"%(self.gold_facet_convex_hull_surface_area) )
        print("Total surface area of edges: %.2f nm^2"%( self.gold_facet_convex_hull_edge_surface_area ) )
        print("Total surface area with convex hull: %.2f nm^2"%( self.self_assembly_structure.surface_area_hull ) )
        print("---- GRAFTING DENSITIES ----")
        print("Total sulfur atoms on surface: %d"%(len(self.sulfur_atom_index_within_cutoff) ))
        print("Total found from self-assembly (should be same as above): %d" %(self.self_assembly_structure.num_gold_sulfur_bonding_per_frame[-1]) )
        print("Total facet surface density: %.2f Angs^2/lig"%(self.planar_density_angs_per_lig))
        print("Total edge surface density: %.2f Angs^2/lig"%(self.planar_density_angs_per_lig_edge))
        return
    
    ### FUNCTION TO FIND THE SURFACE AREA 
    
    
    ### FUNCTION TO FIND GOLD SULFUR COORDINATION GROUPS
    def find_gold_sulfur_coordination_groups(self):
        '''
        The purpose of this function is to look at all the nearby sulfur and surface gold atom, choose the closest gold-sulfur atom, then classify the gold as within a faceted group.
        NOTE:
            - Since group list removes all '-1' classifications, we are ignoring facets that are not represented well
            - This function essentially groups sulfur atoms based purely on the facets
        INPUTS:
            self: [class] 
                self object
        OUTPUTS:
            self.gold_sulfur_coord_split_into_facets_serial_index_dict: [dict]
                dictionary containing the sulfur serial index based on 'self.sulfur_serial_index_within_cutoff' indices
        '''
        ## FINDING DISTANCE MATRIX
        gold_sulfur_coord_distances =  self.sulfur_gold_distances[self.sulfur_serial_index_within_cutoff[:, np.newaxis], self.gold_surface_atom_indices ] # SHAPE: num_sulfur_within_cutoff, num_surface_gold
        
        
        ## FINDING INDICES WHERE WE HAVE REACHED A MINIMUM
        gold_sulfur_coord_closest_gold_atom_index =  self.gold_surface_atom_indices[np.argmin(gold_sulfur_coord_distances, axis=1)] # SHAPE: num_sulfur (CONVERTED TO ATOM INDEX)
        
        ## CONVERTING DICTIONARY TO ATOM INDICES
        gold_facet_groups_based_on_atom_index = convert_each_key_in_dict_given_positions( 
                                                                                        dict_to_convert = self.gold_facet_groups,
                                                                                        conversion_indices= self.gold_facet_atom_indices,
                                                                                        )
            
        ## COMPARING THE LISTS AND GETTING THE SULFUR ATOM POSITIONS
        self.gold_sulfur_coord_split_into_facets_serial_index_dict = find_intersect_group_list_to_atom_index( group_dict = gold_facet_groups_based_on_atom_index,
                                                                                   atom_index_list = gold_sulfur_coord_closest_gold_atom_index
                                                                                  )

        return
    
    ### FUNCTION TO PLOT THE FACET AND SURFACE ATOMS
    def plot3d_sulfur_per_facet_basis(self):
        '''
        The purpose of this function is to color sulfur atoms on top of gold atoms with the same color
        INPUTS:
            self: [class] 
                self object            
        OUTPUTS:
            
        '''
        ## IMPORTING FUNCTIONS
        import matplotlib
        import matplotlib.pyplot as plt
        
        ## DEFINING SULFUR COLORING
        sulfur_style={
                        's'         : 120,
                        'alpha'     : 1,
                        }
        
        ## DEFINING EDGE STYLE
        edge_style= {    'color'      : 'black',
                       's'          : 100,
                       'marker'     : 'o',
                       'label'      : 'edge atoms',
                   }
        
        ## PLOTTING THE COLORED FACETS
        fig, ax =self.plot3d_surface_colored_with_convex_hull()
        # fig, ax = self.plot3d_surface_colored_facets()
#        
#        ## SETTING TITLE
        ax.set_title("Colored sulfur on color-coded facets")
        
#        from MDDescriptors.core.plot_tools import create_3d_axis_plot_labels
#        fig, ax = create_3d_axis_plot_labels( labels = ['x (nm)', 'y (nm)', 'z (nm)'] )
#        
#        ## DRAWING THE EDGES
#        ax.scatter( self.gold_edge_atom_positions[:, 0], self.gold_edge_atom_positions[:, 1], self.gold_edge_atom_positions[:, 2], **edge_style  )
        
        ## LOOPING THORUGH EACH KEY
        for idx, each_group in enumerate(sorted(self.gold_sulfur_coord_split_into_facets_serial_index_dict)):
            ## FINDING THE POSITIONS
            sulfur_positions = self.nearby_sulfur_coordinates[self.gold_sulfur_coord_split_into_facets_serial_index_dict[each_group]]
            ## FINDING COLOR
            current_group_color = self.cmap(idx) # colors[idx]
            ## PLOTTING SCATTER
            ax.scatter( sulfur_positions[:, 0], sulfur_positions[:, 1], sulfur_positions[:, 2], c = current_group_color, 
                       # label = "facet_group (%s)"%(each_group),
                       **sulfur_style )
        
        ## ADDING LEGEND
        # ax.legend(loc = 'right')
        
        return fig, ax
        
    ### FUNCTION TO PLOT THE FACET AND SURFACE ATOMS
    def plot3d_sulfur_color_coded_on_gold(self):
        ''' ***DEPRECIATED***
        The purpose of this function is to plot the sulfur on the gold that is color-coded according to the number of coordinated gold atoms that are facets
        INPUTS:
            self: [class] 
                self object
        OUTPUTS:
             fig, ax: figure and axis for the plot
        REFERENCES:
            Custom scatter plot with matplotlib: https://www.robotswillkillusall.org/posts/mpl-scatterplot-colorbar.html
        '''
        ## IMPORTING FUNCTIONS
        import matplotlib
        import matplotlib.pyplot as plt
        
        ## DEFINING SULFUR COLORING
        sulfur_style={
                        's'         : 120,
                        'alpha'     : 1 ,
                        }
        
        ## DEFINING EDGE STYLE
        edge_style= {    'color'      : 'black',
                       's'          : 100,
                       'marker'     : 'o',
                       'label'      : 'edge atoms',
                   }
        
        ## PLOTTING THE COLORED FACETS
        # fix, ax =self.plot3d_surface_colored_with_convex_hull()
        # fig, ax = self.plot3d_surface_colored_facets()
        
        from MDDescriptors.core.plot_tools import create_3d_axis_plot_labels
        fig, ax = create_3d_axis_plot_labels( labels = ['x (nm)', 'y (nm)', 'z (nm)'] )
        
        ## DRAWING THE EDGES
        ax.scatter( self.gold_edge_atom_positions[:, 0], self.gold_edge_atom_positions[:, 1], self.gold_edge_atom_positions[:, 2], **edge_style  )
        
        ## GETTING CMAP INFORMATION
        cmap_color = matplotlib.cm.get_cmap('viridis')
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=1) # Normalize between 0 and 1
        colors = [cmap_color(normalize(value)) for value in self.ratio_of_facet_surface_atoms]
        
        ## PLOTTING
        ax.scatter(self.nearby_sulfur_coordinates[:, 0], 
                   self.nearby_sulfur_coordinates[:, 1], 
                   self.nearby_sulfur_coordinates[:, 2], color =  colors, **sulfur_style ) # label = "Ratio: %.2f"%(each_assignment), 
        
        ## ADDING LEGEND
        ax.legend(loc = 'lower right')
        
        ## ADDING COLOR BAR 
        cax, _ = matplotlib.colorbar.make_axes(ax)
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap_color, norm=normalize)
        return fig, ax
    
    ### FUNCTION TO PLOT THE SULFUR ON THE GOLD
    def plot3d_sulfur_on_gold(self):
        '''
        The purpose of this function is to plot the sulfur on the gold atom
        INPUTS:
            self: [class] 
                self object
        OUTPUTS:
             fig, ax: figure and axis for the plot
        '''
        ## DEFINING SULFUR COLORING
        sulfur_style={
                        'color'     : 'yellow',
                        's'         : 120,
                        }

        
        ## PLOTTING THE COLORED FACETS
        fig, ax = self.plot3d_surface_colored_facets()
        
        ## SETTING TITLE
        ax.set_title("Sulfur atoms on the color-coded faceted gold atoms")
        
        ## PLOTTING SULFUR POSITIONS AS A SCATTER
        ax.scatter( self.nearby_sulfur_coordinates[:, 0], self.nearby_sulfur_coordinates[:, 1], self.nearby_sulfur_coordinates[:, 2], label = "sulfur_atoms", **sulfur_style )
        
        ## ADDING LEGEND
        ax.legend(loc = 'lower right')
        
        return fig, ax
    
    ### FUNCTION TO CALCULATE GOLD-SULFUR BONDING
    def calc_sulfur_gold_bonding_distances(self, traj, periodic=True):
        '''
        The purpose of this function is to find sulfur-gold distances
        INPUTS:
            self: [class] 
                self object
            traj: [class]
                trajectory from md.traj
            periodic: [logical, default=True]
                True if you want periodic boundary conditions applied
        OUTPUTS:
            self.sulfur_atom_index: [np.array, shape=(num_sulfur,1)]    
                sulfur index of all sulfur atoms
            self.gold_atom_index: [np.array, shape=(num_sulfur,1)]    
                gold atom index based on optimization of gold index of self_assembly_structure
            self.sulfur_gold_distances: [np.array, shape=(num_sulfur, num_gold)]
                pair distances between sulfur and gold atoms
        '''
        ## CALCULATING SULFUR GOLD BONDING DISTANCES
        self.sulfur_gold_distances = calc_tools.calc_pair_distances_between_two_atom_index_list( 
                                                                    traj = traj[self.frame], 
                                                                    atom_1_index = self.sulfur_atom_index,
                                                                    atom_2_index = self.gold_atom_index,
                                                                    periodic = periodic,
                                                                    )[0] # Escaping the frame
        return
    
    ### FUNCTION TO FIND THE SULFUR INDICES 
    def find_nearby_sulfur_indices(self):
        '''
        The purpose of this function is to get the closest sulfur atoms
        INPUTS:
            self: [class] 
                self object
        OUTPUTS:
            self.sulfur_serial_index_within_cutoff: [np.array, shape=(num_sulfur, 1)]
                sulfur serial index referenced to sulfur gold distances
            self.sulfur_index_within_cutoff: [np.array]
                sulfur atom index referenced to the trajectory that is within the cutoff
        '''
        ## FINDING NEARBY SULFUR INDICES
        self.sulfur_serial_index_within_cutoff = np.where(np.any(self.sulfur_gold_distances < self.gold_sulfur_cutoff,axis=1))[0]
         
        ## RETURNS SULFUR INDICES WITHIN THE CUTOFF
        self.sulfur_atom_index_within_cutoff = self.sulfur_atom_index[self.sulfur_serial_index_within_cutoff]
        return
        
    
    ### FUNCTION TO FIND THE DISTANCE MATRIX BETWEEN GOLD ATOMS FOR A FACET
    def calc_gold_facet_distance_matrix(self, traj, periodic=True):
        '''
        The purpose of this function is to find the unique facets based on atom indicies and cutoff
        INPUTS:
            self: [class] 
                self object
            traj: [class]
                trajectory from md.traj
            periodic: [logical, default=True]
                True if you want periodic boundary conditions applied
        OUTPUTS:
            self.gold_facet_distance_matrix: [np.array, shape=(num_facets, num_facets)]
                symmetric distance matrix for gold facet atoms
        '''
        ## CREATING ATOM PAIRS FOR THE FACET ATOM
        gold_facet_atom_pairs, upper_triangular = calc_tools.create_atom_pairs_with_self(self.gold_facet_atom_indices)
        ## FINDING TOTAL FACET ATOMS
        total_facet_atoms = len(self.gold_facet_atom_indices)
        ## FINDING DISTANCES FOR A SPECIFIC FRAME
        gold_facet_distances = md.compute_distances( traj = traj[self.frame], 
                                                         atom_pairs = gold_facet_atom_pairs,
                                                         periodic = periodic
                                                         )
        ## CREATING DISTANCE MATRIX
        self.gold_facet_distance_matrix = np.zeros( (total_facet_atoms, total_facet_atoms)  )
        ## INPUTTING TO DISTANCE MATRIX
        self.gold_facet_distance_matrix[upper_triangular] = gold_facet_distances
        ## FINDING THE FULL DISTANCE MATRIX BASED ON SYMMETRY
        self.gold_facet_distance_matrix = self.gold_facet_distance_matrix + self.gold_facet_distance_matrix.T
        
        return

    ### FUNCTION TO CALCULATE THE FACET CONVEX HULL OF ENTIRE SURFACE
    def calc_gold_surface_convex_hull(self, traj):
        '''
        The purpose of this function is to calculate the convex hull based on entire surface
        INPUTS:
            traj: [object]
                trajectory from md.traj
        OUTPUTS:
            self.gold_surface_convex_hull: [object]
                convex hull from ConvexHull function of scipy.spatial
            self.gold_surface_convex_hull_surface_area: [float]
                surface area of entire gold particle
        '''
        ## IMPORING ConvexHull command
        from scipy.spatial import ConvexHull
        
        ## DEFINING GOLD INDICES (SURFACE)
        # surface_gold_index = self.gold_atom_index[self.gold_surface_atom_indices] 
        
        ## FINDING GOLD COORDINATES FOR LAST FRAME
        # points = traj.xyz[self.frame, surface_gold_index, : ]
        
        ## GETTING CONVEX HULL
        # self.gold_surface_convex_hull = ConvexHull(points = points)
        self.gold_surface_convex_hull = ConvexHull(points = self.gold_edge_atom_positions)
        
        ## FINDING AREA
        self.gold_surface_convex_hull_surface_area = self.gold_surface_convex_hull.area
        

    ### FUNCTION TO GET CONVEX HULL OF EACH OF THE POINTS
    def calc_gold_facet_convex_hull(self):
        '''
        The purpose of this function is to get the convex hull of each facet
        INPUTS:
            self: [class] 
                self object
        OUTPUTS:
            self.gold_facet_convex_hull: [list]
                convex hull for each facet based on a sorted version of self.gold_facet_groups
            self.gold_facet_convex_hull_surface_area_per_facet: [list]
                list of surface areas per facet
            self.gold_facet_convex_hull_surface_area: [float]
                total surface area of all facets via convex hull
        '''
        ## IMPORING ConvexHull command
        from scipy.spatial import ConvexHull
        
        ## DEFINING EMPTY LIST TO STORE THE HULL
        self.gold_facet_convex_hull = []
        
        ### LOOPING THROUGH EACH GROUP
        for idx, each_group in enumerate(sorted(self.gold_facet_groups)):
            ## DEFINING CURRENT INDICES
            gold_facet_atom_indices = self.gold_facet_groups[each_group]
            ## DEFINING CURRENT POSITIONS
            gold_facet_atom_positions = self.gold_facet_atom_positions[gold_facet_atom_indices]
            ## CALCULATING THE HULL
            self.gold_facet_convex_hull.append(ConvexHull(points = gold_facet_atom_positions))
        
        ## FINDING TOTAL SURFACE AREA
        self.gold_facet_convex_hull_surface_area_per_facet = np.array([ each_hull.area for each_hull in self.gold_facet_convex_hull])/ 2.0 # Divide by two since we have a plane. We do not want to double-count!
        self.gold_facet_convex_hull_surface_area = np.sum(self.gold_facet_convex_hull_surface_area_per_facet)
        
        return

    ### FUNCTION TO PLOT FACET AND EDGES WITH THE FACET COLORED
    def plot3d_surface_colored_facets(self):
        '''
        The purpose of this function is to plot surface facets versus edges and color the facets accordingly
        INPUTS:
            self: [class] 
                self object
        OUTPUTS:
            xyz plot of facet and edges
            fix, ax -- figure and axis for your plot
            cmap: [list]
                list of colors used for the plotting
        '''
        ## IMPORTING GLOBAL VARIABLES
        from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
        from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot
        
        ## CREATING PLOT
        fig, ax = create_3d_axis_plot()
        
        ## DEFINING TITLE
        ax.set_title("Color coded facets and edges")
        
        ## DEFINING LABELS
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_zlabel("z (nm)")
        
        ## SETTING DICTIONARY FOR EACH TYPE OF ATOM
        plot_dict = {
                      'facet':
                          {    
                               's'          : 100,
                               'marker'     : 'o',
                                  },
                      'edge':
                          {    'color'      : 'black',
                               's'          : 100,
                               'marker'     : 'o',
                               'label'      : 'edge atoms',
                                  }
                        }
        
        ## DRAWING THE EDGES
        ax.scatter( self.gold_edge_atom_positions[:, 0], self.gold_edge_atom_positions[:, 1], self.gold_edge_atom_positions[:, 2], **plot_dict['edge']  )
        
        ## DRAWING EACH FACET
        for idx, each_group in enumerate(sorted(self.gold_facet_groups)):
            ## DEFINING CURRENT INDICES
            gold_facet_atom_indices = self.gold_facet_groups[each_group]
            ## DEFINING CURRENT POSITIONS
            gold_facet_atom_positions = self.gold_facet_atom_positions[gold_facet_atom_indices]
            ## DEFINING CURRENT COLOR
            current_group_color = self.cmap(idx) # colors[idx]
            ## PLOTTING SCATTER
            ax.scatter( gold_facet_atom_positions[:, 0], gold_facet_atom_positions[:, 1], gold_facet_atom_positions[:, 2], c = current_group_color, 
                       label = "facet_group (%s)"%(each_group),
                       **plot_dict['facet'] )
        
        ## DRAWING LEGEND
        ax.legend(loc = 'lower right')
        
        return fig, ax
    
    ### FUNCTION TO PLOT THE FACET AS A CONVEX HULL
    def plot3d_surface_colored_with_convex_hull(self):
        '''
        The purpose of this function is to color the surface with the convex hull
        INPUTS:
            self: [class] 
                self object
        OUTPUTS:
            fig, ax: figure and axis for plot
        '''
        
        ## DEFINING CONVEX HULL INFORMATION
        line_styles={
                    'linestyle' :   '-' ,
                    'linewidth' :   2   ,
                     }
        
        ## CREATING FIGURE
        # fig, cmap = self.plot3d_surface_colored_facets()
        from MDDescriptors.core.plot_tools import create_3d_axis_plot_labels
        fig, ax = create_3d_axis_plot_labels( labels = ['x (nm)', 'y (nm)', 'z (nm)'] )
        
        ## SETTING TITLE
        ax.set_title("Color-coded convex hull surface areas")
        
        ## ADDING TO THE FIGURE WITH CONVEX HULL
        for idx, each_group in enumerate(sorted(self.gold_facet_groups)):
            ## DEFINING CURRENT HULL
            hull = self.gold_facet_convex_hull[idx]
            ## DEFINING CURRENT COLOR
            current_hull_color=self.cmap(idx)
            ax = plot_convex_hull_color( hull = hull,
                                         ax = ax,
                                         color = current_hull_color,
                                         label = 'facet_group (%d), SA: %.1f nm^2'%(idx, hull.area),
                                         **line_styles
                                        )
        ## DRAWING LEGEND
        ax.legend(loc = 'lower right')
        
        return fig, ax
            

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON
    # analysis_dir=r"180328-ROT_SPHERICAL_TRANSFER_LIGANDS_6nm" # Analysis directory
    analysis_dir=r"SELF_ASSEMBLY_FULL" # Analysis directory
    # category_dir="spherical" # category directory
    category_dir="EAM" # category directory spherical
    # specific_dir="spherical_6_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    diameter="5"
    specific_dir= category_dir + "_"+ diameter +"_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol_Trial_1" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # 180328-ROT_SPHERICAL_TRANSFER_LIGANDS_6nm/spherical
    ### DEFINING PATH
    path2AnalysisDir=r"R:\scratch\nanoparticle_project\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"gold_ligand_equil.gro" # Structural file
    xtc_file=r"gold_ligand_equil_whole.xtc" # Trajectory file
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    #%%
    from MDDescriptors.application.nanoparticle.self_assembly_structure import self_assembly_structure
    ### DEFINING INPUT DATA
    input_details = {   'traj_data'          :           traj_data,                      # Trajectory information
                         'frame'             :                  -1,                     # frame to study self assembly structure
                         'save_disk_space'   :          False    ,                        # Saving space
                         'gold_sulfur_cutoff':      GOLD_SULFUR_CUTOFF, # Gold sulfur cutoff
                         'maximum_num_gold_atoms':         1 , # 3,
                         'min_gold_samples':                4,          # Must be greater than one
                         'analysis_classes'  : [[ self_assembly_structure, '180814-FINAL' ], ## REUSING SELF ASSEMBLY STRUCTURE CLASS
                                                ],
                         'pickle_loading_file': specific_dir,       ## SPECIFIC DIRECTORY FOR LOADING
                         'debug'             :          True, # Debugging 
                         'want_expand_edges' :  True,   
                         }
    import matplotlib.pyplot as plt
    plt.close('all')
    ### RUNNING NANOPARTICLE GOLD STRUCTURE
    self_assembly_coord = self_assembly_sulfur_gold_coordination( **input_details )

    
    #%%
    
    ## DEFINING TRAJECTORY
    traj = traj_data.traj
    

    
    ## DEFINING GOLD GOLD CUTOFF
    gold_gold_cutoff = self_assembly_coord.gold_gold_cutoff
    
    
    ## DEFINING GOLD ATOM INDEX
    gold_atom_index = self_assembly_coord.gold_atom_index
    
    ## DEFINING GOLD SURFACE ATOM INDEX
    gold_surface_atom_indices = self_assembly_coord.gold_surface_atom_indices
    
    ## DEFINING GOLD FACET ATOM INDICES
    gold_facet_atom_indices = self_assembly_coord.gold_facet_atom_indices
    
    ## DEFINING GOLD EDGE ATOM INDICES
    gold_edge_atom_indices = self_assembly_coord.gold_edge_atom_indices
    
    ## DEFINING FRAME
    frame = self_assembly_coord.frame
    

        
    
    
    
    #%%
    
    
    ## IMPORING ConvexHull command
    from scipy.spatial import ConvexHull
    
    points=[ 
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
            ]
    
    hull = ConvexHull(points = points
            
            )
    
    
    
    
    #%%
    
    plot_convex_hull_color(self_assembly_coord.gold_surface_convex_hull)
    

    #%%
    
    self_assembly_coord.plot3d_sulfur_per_facet_basis()

    #%%
    
    ### FINDING THE EDGE SURFACE AREA
    
    
    


    #%%
    
    

    
    
    ## FINDING DISTANCE MATRIX
    gold_sulfur_coord_distances =  self_assembly_coord.sulfur_gold_distances[self_assembly_coord.sulfur_serial_index_within_cutoff[:, np.newaxis], self_assembly_coord.gold_surface_atom_indices ] # SHAPE: num_sulfur_within_cutoff, num_surface_gold
    
    ## FINDING INDICES WHERE WE HAVE REACHED A MINIMUM
    gold_sulfur_coord_closest_gold_serial_index = self_assembly_coord.gold_surface_atom_indices[np.argmin(gold_sulfur_coord_distances, axis=1)] # SHAPE: num_sulfur
    
    ## DEFINING FACET GROUPS
    gold_facet_groups = self_assembly_coord.gold_facet_groups
        
    ## CONVERTING DICTIONARY TO ATOM INDICES
    gold_sulfur_coord_split_into_facets_atom_index_dict = convert_each_key_in_dict_given_positions( 
                                                                                                        dict_to_convert = gold_facet_groups,
                                                                                                        conversion_indices= self_assembly_coord.gold_facet_atom_indices,
                                                                                                        )
    
    ## COMPARING THE LISTS AND GETTING THE SULFUR ATOM POSITIONS
    gold_sulfur_coord_split_into_facets_dict = find_intersect_group_list_to_atom_index( group_dict =gold_sulfur_coord_split_into_facets_atom_index_dict, # gold_facet_groups,
                                                                               atom_index_list = gold_sulfur_coord_closest_gold_serial_index
                                                                              )
    
    #%%
    
    
    
    
    
    
    
    
    
    ## FINDING THE CLOSEST GOLD ATOM SURFACE INDEX
    # gold_sulfur_coord_closest_gold_atom_index = self_assembly_coord.gold_surface_atom_indices[gold_sulfur_coord_closest_gold_serial_index]
    # RETURNS SULFUR COORDINATE WITH THE CLOSEST ATOM INDEX
    
    ### SORTING INDICES BASED ON WHETHER OR NOT THE SULFUR IS ON A FACET ATOM
    ## FINDING INDICES WHERE YOUR NEARBY GOLD ATOM IS A FACET ATOM
    indices_sulfur_to_facet_gold_atoms =  np.isin(gold_sulfur_coord_closest_gold_atom_index, self_assembly_coord.gold_facet_atom_indices )
    #%%
    ## FINDING ALL FACET INDICES
    # gold_sulfur_coord_closest_facet_atom_indices=np.isin( self_assembly_coord.gold_facet_atom_indices, gold_sulfur_coord_closest_gold_atom_index[indices_sulfur_to_facet_gold_atoms] ).nonzero()
    gold_sulfur_coord_closest_facet_atom_indices=gold_sulfur_coord_closest_gold_atom_index[indices_sulfur_to_facet_gold_atoms]
    ## DEFINING FACET GROUPS
    gold_facet_groups = self_assembly_coord.gold_facet_groups
    
    ## CONVERTING DICTIONARY TO ATOM INDICES
    gold_sulfur_coord_split_into_facets_atom_index_dict = convert_each_key_in_dict_given_positions( 
                                                                                                        dict_to_convert = gold_facet_groups,
                                                                                                        conversion_indices= self_assembly_coord.gold_atom_index,
                                                                                                        )
    
    
    #%%

        
    ## CREATING INDEXES ON FACETS
    gold_sulfur_coord_split_into_facets_dict = find_intersect_group_list_to_atom_index( group_dict = gold_facet_groups,
                                                                                   atom_index_list = gold_sulfur_coord_closest_facet_atom_indices
                                                                                  )
    #%%
    ## CONVERTING DICTIONARY TO ATOM INDICES
    gold_sulfur_coord_split_into_facets_atom_index_dict = convert_each_key_in_dict_given_positions( 
                                                                                                        dict_to_convert = gold_sulfur_coord_split_into_facets_dict,
                                                                                                        conversion_indices= self_assembly_coord.sulfur_serial_index_within_cutoff[indices_sulfur_to_facet_gold_atoms]
                                                                                                        )
    
    #%%
    
    test = self_assembly_coord.nearby_sulfur_coordinates[self_assembly_coord.gold_sulfur_coord_split_into_facets_serial_index_dict['0']]
    # test[:, 0]
    
    #%%
    
    
    
    np.intersect1d(self_assembly_coord.gold_facet_atom_indices, gold_sulfur_coord_closest_facet_atom_indices).shape
    
    #%%
    
    np.argmin(self_assembly_coord.gold_sulfur_coord_distances, axis=1)
    #%%
    print(np.sum([  self_assembly_coord.gold_sulfur_coord_split_into_facets_serial_index_dict[each_key][0].size for each_key in self_assembly_coord.gold_sulfur_coord_split_into_facets_serial_index_dict]))
    
    #%%
    print(np.sum([  gold_sulfur_coord_split_into_facets_dict[each_key][0].size for each_key in gold_sulfur_coord_split_into_facets_dict]))
    
    
    
    #%%
    gold_atom_index = self_assembly_coord.gold_atom_index
    gold_atom_index_surface = self_assembly_coord.gold_sulfur_coord_gold_indices
    test = np.intersect1d(gold_atom_index, gold_atom_index_surface)
    # sulfur_serial_indices = self_assembly_coord.sulfur_serial_index_within_cutoff
    
    
    #%%
    self_assembly_coord.surface_gold_sulfur_distances.shape
    
    
    ## FINDING SURFACE FACET INDICES THAT ARE NOT UNASSIGNED
    gold_facet_labels_without_unassigned = self_assembly_coord.gold_facet_atom_indices[np.where(self_assembly_coord.gold_facet_labels != -1)[0]]
    
    # self_assembly_coord.gold_facet_atom_indices[
    ## FINDING SURFA
    
    