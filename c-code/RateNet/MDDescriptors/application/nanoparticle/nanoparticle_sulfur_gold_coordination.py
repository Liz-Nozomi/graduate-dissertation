# -*- coding: utf-8 -*-
"""
nanoparticle_sulfur_gold_coordination.py
The purpose of this script is to find how many gold atoms are coordinated around the sulfur atom. 

Written by: Alex K. Chew (alexkchew@gmail.com, 06/29/2018)

CLASSES:
    nanoparticle_sulfur_gold_coordination: finds nanoparticle gold-sulfur coordination in terms of facets versus edges

FUNCTIONS:
    find_nearest_gold_atoms_to_sulfur: finds gold atom closest to sulfur atoms

** UPDATES **
20180702 - AKC - completed draft of script
20180718 - AKC - Fixed bug in script with reordering of the distance matrix
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

### FUNCTION TO FIND THE NEAREST GOLD ATOMS
def find_nearest_gold_atoms_to_sulfur(distances, maximum_num_gold_atoms = 3):
    '''
    The purpose of this function is to find the indexes of the nearest gold atoms given a distance matrix such that:
        - shape: num_frames x num_gold_atoms x num_sulfur_atoms
    INPUTS:
        distances: [np.array, shape=(num_frames, num_gold_atoms, num_sulfur_atoms)] distance matrix between gold and sulfur
        maximum_num_gold_atoms: [int, default=3] number of maximum gold atoms to search for
    OUTPUTS:
        indices_of_nearest: [np.array, shape=(num_frames, maximum_num_gold_atoms, num_sulfur_atoms)] Indices of the gold atoms that are most closest to the sulfur at eachframe
    '''
    ## SORTING THE DISTANCES
    sorted_distances = np.argsort(distances, axis = 1)
    # RETURNS: num_frames, num_gold, num_sulfur.
    # The indices are sorted along the sulfur index. That way, the first index is the smallest
    indices_of_nearest = sorted_distances[:, 0:maximum_num_gold_atoms, :]
    # RETURNS: num_frames, max_num_gold_atoms, num_sulfur
    return indices_of_nearest

#### FUNCTION TO FIND A FUNCTION FROM A LIST
#def find_function_name_from_list(function_list, class_name):
#    '''
#    The purpose of this function is to find a class from a list
#    INPUTS:
#        function_list: [list] list of classes/functions
#        class_name: [str] name of the class you are interested in
#    OUTPUTS:
#        class_of_interest: [class] class/function object with the name you are interested in
#    '''
#    ## DO A LOOP TO TRY TO FIND THE CLASS NAME
#    class_of_interest = [ each_function for each_function in function_list if each_function.__name__ == class_name][0]
#    return class_of_interest
#
#### FUNCTION TO CONVERT PICKLE FILE NAMES BASED ON YOUR DESIRED SETTINGS
#def convert_pickle_file_names(pickle_file_name, conversion_type = None):
#    '''
#    The purpose of this function is to convert the pickle file name so you can correctly load the pickle. This is useful if you have different nomenclature for different simulation setups.
#    INPUTS:
#        pickle_file_name: [str]
#            name of the pickle file you are running
#        conversion_type: [str, default=None]
#            conversion type that you want. The list of conversions are below:
#                'spherical_np_to_self_assembled_structure': converts nanoparticle systems to self assembled structure
#    OUTPUTS:
#        updated_pickle_file_name: [str]
#            name of the updated pickle file
#    '''
#    from MDDescriptors.core.decoder import decode_name
#    if conversion_type == None:
#        updated_pickle_file_name = pickle_file_name
#    ## THIS CONVERSION TYPE CONVERTS SPHERICAL NP RUNS TO SELF ASSEMBLED STRUCTURE NOMENCLATURE
#    # Ex: EAM_310.15_K_2_nmDIAM_dodecanethiol_CHARMM36_Trial_1 -> EAM_2_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol_Trial_1
#    elif conversion_type == "spherical_np_to_self_assembled_structure":
#        ## USING DECODING FUNCTIONS
#        decoded_name = decode_name(pickle_file_name, decode_type="nanoparticle")
#        ## UPDATING PICKLE FILE NAME, e.g. EAM_2_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol_Trial_2
#        updated_pickle_file_name = "%s_%d_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol_Trial_%d"%(decoded_name['shape'], decoded_name['diameter'], decoded_name['trial'] ) 
#        
#    return updated_pickle_file_name
#
#
#### FUNCTION TO LOAD PICKLE BASED ON ANALYSIS CLASSES AND STRING
#def load_pickle_for_analysis( analysis_classes, function_name, pickle_file_name, conversion_type = None, current_work_dir = None):
#    '''
#    The purpose of this function is to re-load a pickle for subsequent analysis. We will use the data for this pickle to compute new information, then save the entire thing under multi_traj analysis tools
#    It is expected that this function is run several times to load specific information from a pickle
#    INPUTS:
#        analysis_classes: [list of list]
#            list of list of analysis classes, e.g. [[ self_assembly_structure, '180814-FINAL' ], ...],
#            Note: the first entry is the class function, the second entry is the date/location for loading the pickle.
#        function_name: [function]
#            name of the function you want to load.
#        pickle_file_name: [str]
#            string of the pickle file name that you want to load. This should be located within function_name > date > pickle_file_name
#        current_work_dir: [str]
#            path to directory that has the pickle folder
#    OUTPUTS:
#        multi_traj_results: [class]
#            analysis tool based on your inputs
#    '''
#    ## FINDING FUNCTION FROM THE LIST
#    specific_analysis_class = [each_analysis for each_analysis in analysis_classes if each_analysis[0].__name__ == function_name ][0]
#    
#    ## CONVERSION OF PICKLE FILE NAME IF NEEDED
#    updated_pickle_file_name = convert_pickle_file_names( pickle_file_name = pickle_file_name,
#                                                          conversion_type = conversion_type,
#                                                         )
#    ## RUNNING MULTI TRAJ PICKLING
#    multi_traj_results = load_multi_traj_pickle(specific_analysis_class[1], specific_analysis_class[0], updated_pickle_file_name, current_work_dir = current_work_dir )
#    
#    return multi_traj_results
    
###############################################################
### CLASS FUNCTION TO ANALYZE THE GOLD-SULFUR COORDINATIONS ###
###############################################################
class nanoparticle_sulfur_gold_coordination:
    '''
    The purpose of this function is to find the number of gold atoms coordinated around the sulfur atoms.
    INPUTS:
        traj_data: Data taken from import_traj class
        pickle_loading_file: [str]
            name of the pickle loading file (currently)
        analysis_classes: [list]
            list of list of analysis classes and corresponding dates to load the pickles
        ligand_names: [list] list of ligand residue names. Note that you can list as much as you'd like, we will check with the trajectory to ensure that it is actually there.
        itp_file: itp file name (note! We will look for the itp file that should be in the same directory as your trajectory)
            if 'match', it will look for all itp files within the trajectory directory and locate all itp files that match your ligand names.
                NOTE: This is currently assuming homogenous ligands -- this will need to be adjusted for multiple ligands
        split_traj: [int] number of times you want to split the trajectory to run this analysis tool
        gold_sulfur_cutoff: [float] cutoff between gold and sulfur
        coord_num_surface_to_bulk_cutoff: [int] coordination number between surface and bulk atoms. Everything less than or equal to this number is considered a surface atom
        separated_ligands: [logical, default=False] True if your ligands are not in the same itp file
        gold_atom_name: [str, default='Au'] name of gold -- used to find gold atom indices
        debug: [logical, default=False] True if you want to see a plot of the surface atoms
        save_disk_space: [logical, Default: True] True if you want to save disk space by removing the following variables
            self.gold_sulfur_atom_pairs, self.gold_surface_facet_vs_edge_classification
    OUTPUTS:
        ## STORING INPUTS
            self.gold_atom_name: [str, default='Au'] name of gold -- used to find gold atom indices
            self.gold_sulfur_cutoff: [float] cutoff between gold and sulfur
            
        ---- from self assembly structure ---
        ## GOLD INFORMATION
            self.gold_gold_cutoff: [float] cutoff for gold-gold nearest neighbors calculation
            self.gold_coord_num: numpy stacked version of: 
                    index: [np.array, shape=(Nx1)] atom index of the gold atoms
                    coordination_number: [np.array, shape=(Nx1)] number of gold atoms coordinated for the index
        ## GOLD SURFACE INFORMATION
            self.gold_surface_indices: [np.array] array with the indices of the surface atoms
            self.gold_surface_atom_index: [np.array] array with gold-surface atom indices (referenced to traj.topology)
            self.gold_surface_coord_num: [np.array, shape=(num_surface_atoms)] surface coordination numbers
            self.gold_surface_num_atom: [int] total number of surface atoms
        ## FACETS AND EDGES
            self.gold_surface_facet_vs_edge_classification: [np.array, shape=(num_surface_atoms, 1)] Classification of edge and surface, where:
                Values of 0 indicate edges
                Values of 1 indicate facets
            self.gold_surface_facet_index:  [np.array, shape=(num_facets, 1)]   Indices of gold surface atoms that are faceted
            self.gold_surface_edge_index:   [np.array, shape=(num_edge, 1)]     Indices of gold surface atoms that are on the edge
            
        ## RESULTS
            self.ratio_of_facet_surface_atoms: [np.array, shape=(num_frames, num_sulfur_atoms)] ratio of facet atoms to total surrounding atoms
                - can contain nan's. If this is the case, then the sulfur atom is not nearby any of the gold atoms based on your cutoff.
                - all numpy nan's will be ignored, so no error message will be displayed
        
    FUNCTIONS:
        clean_disk: cleans up disk space
        find_facets_and_edges_based_on_coord_num: finds facets and edges based on coordination number
        plot3d_surface_facets_vs_edges: plots 3D lattice of the surface facets versus edges
        calc_facets_to_edges_ratio: [staticmethod] calculates facet to total nearby surface atoms for each sulfur atom and each frame
        
    ACTIVE FUNCTIONS:
        plot_gold_coord_num: plots gold coordination number
        plot_hist_ratio_of_facet_surface_atoms: plots histogram of facet vs. total surface atoms
        
    ALGORITHM:
        - Locate all gold indices
        - Find coordination number of all gold atoms (using last frame of gold atoms)
            - Plot the coordination number of gold (distribution)
        - Identify the groups for the gold:
            - surface atoms
                - edge atoms
                - facet atoms
            - bulk atoms
        - Find all sulfur atoms
        - Find distances between all sulfur atoms and gold surface atoms
        - For each sulfur:
            - find all gold atoms within a cutoff
    
    '''
    ### INITIALIZING
    def __init__(self, traj_data, pickle_loading_file, analysis_classes, ligand_names, itp_file, split_traj = 10, 
                 gold_sulfur_cutoff = GOLD_SULFUR_CUTOFF, maximum_num_gold_atoms = 3,
                 separated_ligands = False, gold_atom_name="Au", save_disk_space = True, debug = False):
        
        ### STORING INITIAL VARIABLES
        self.gold_atom_name = gold_atom_name
        self.gold_sulfur_cutoff = gold_sulfur_cutoff
        self.maximum_num_gold_atoms = maximum_num_gold_atoms
        
        ### GETTING ONLY ANALYSIS CLASSES
        self.self_assembly_structure = load_pickle_for_analysis(
                                                                analysis_classes = analysis_classes, 
                                                                function_name = 'self_assembly_structure', 
                                                                pickle_file_name = pickle_loading_file,
                                                                conversion_type = "spherical_np_to_self_assembled_structure",
                                                                current_work_dir = NP_WORKING_DIR,
                                                                )
        
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__))
        
        ### CALCULATING NANOPARTICLE STRUCTURE
        self.structure_np = nanoparticle_structure(traj_data           = traj_data,                # trajectory data
                                                ligand_names        = ligand_names,        # ligand names
                                                itp_file            = itp_file,                 # defines the itp file
                                                structure_types      = None,                     # checks structural types
                                                separated_ligands    = separated_ligands    # True if you want separated ligands 
                                                )
        
        ## DEFINING TRAJECTORY
        traj = traj_data.traj
        
        ##########################################
        ### FINDING INFORMATION ABOUT THE GOLD ###
        ##########################################
        
        ## FINDING ATOM INDEX OF GOLD
        self.gold_atom_index = np.array(calc_tools.find_atom_index(   traj = traj,                    # trajectory
                                                             atom_name = self.gold_atom_name,    # gold atom name
                                                             ))
        
        ## FINDING TOTAL GOLD ATOMS
        self.gold_num_atoms = len(self.gold_atom_index)
        
        ## FINDING COORDINATION NUMBER AND SURFACE ATOMS
        self.gold_coord_num, self.gold_gold_cutoff, self.gold_surface_indices, \
        self.gold_surface_atom_index, self.gold_surface_coord_num, self.gold_surface_num_atom = self.self_assembly_structure.gold_gold_coordination['gold_coord_num'], \
                                                                                                self.self_assembly_structure.gold_gold_coordination['gold_gold_cutoff'], \
                                                                                                self.self_assembly_structure.gold_gold_coordination['gold_surface_indices'], \
                                                                                                self.self_assembly_structure.gold_gold_coordination['gold_surface_atom_index'], \
                                                                                                self.self_assembly_structure.gold_gold_coordination['gold_surface_coord_num'], \
                                                                                                self.self_assembly_structure.gold_gold_coordination['gold_surface_num_atom']
        
        
        ## DETERMINE WHICH SURFACE ATOMS ARE EDGE AND FACET ATOMS
        self.gold_surface_facet_vs_edge_classification, self.gold_surface_facet_index, self.gold_surface_edge_index = \
                                        self.self_assembly_structure.gold_gold_surface_facet_edge_dict['gold_surface_facet_vs_edge_classification'],  \
                                        self.self_assembly_structure.gold_gold_surface_facet_edge_dict['gold_surface_facet_index'],  \
                                        self.self_assembly_structure.gold_gold_surface_facet_edge_dict['gold_surface_edge_index'],
        # self.find_facets_and_edges_based_on_coord_num()
        
        ## PLOTTING SURFACE FACETS AND EDGES TO ENSURE CORRECTNESS
        if debug == True:
            self.plot3d_surface_facets_vs_edges(traj = traj)
        
        ########################################
        ### FINDING INFORMATION ABOUT SULFUR ###
        ########################################
        
        ## ATOM INDICES OF SULFUR
        self.sulfur_atom_index = np.array(self.structure_np.head_group_atom_index)
        
        #############################################
        ### FINDING INFORMATION ABOUT GOLD-SULFUR ###
        #############################################
        
        ## GETTING ATOM PAIRS BETWEEN GOLD AND SULFUR
        self.gold_sulfur_atom_pairs = calc_tools.create_atom_pairs_list( atom_1_index_list = self.gold_atom_index[self.gold_surface_atom_index],
                                                                         atom_2_index_list = self.sulfur_atom_index,)
        
        ## FINDING GOLD-SULFUR FACETS TO EDGE RATIO
        # DEFINING INPUTS
        inputs = { 'gold_sulfur_atom_pairs' : self.gold_sulfur_atom_pairs,
                   'gold_atom_index'        : self.gold_surface_atom_index,
                   'sulfur_atom_index'      : self.sulfur_atom_index,
                   'gold_surface_edge_index': self.gold_surface_edge_index,
                   'gold_surface_facet_index': self.gold_surface_facet_index,
                   'gold_sulfur_cutoff'     : self.gold_sulfur_cutoff, ## GOLDSULFUR CUTOFF IS DEPRECIATED
                   'maximum_num_gold_atoms' : self.maximum_num_gold_atoms,
                   'gold_surface_facet_vs_edge_classification': self.gold_surface_facet_vs_edge_classification,
                  
                  }
        self.ratio_of_facet_surface_atoms = calc_tools.split_traj_function( traj = traj,
                                                                            split_traj = split_traj,
                                                                            input_function = self.calc_facets_to_edges_ratio,
                                                                            optimize_memory = True,
                                                                            **inputs
                                                                            )
        ## CLEANING UP DISK
        self.clean_disk( save_disk_space = save_disk_space)
        
        return
    
    
    ### FUNCTION TO CLEAN UP DISK
    def clean_disk(self, save_disk_space = True):
        ''' 
        This function cleans up disk space 
        INPUTS:
            save_disk_space: [logical, Default: True] True if you want to save disk space by removing the following variables
        '''
        if save_disk_space == True:
            self.gold_sulfur_atom_pairs = []
            # self.gold_surface_facet_vs_edge_classification = []
        return
    
    ### FUNCTION TO PLOT GOLD COORDINATION NUMBERS
    def plot_gold_coord_num(self):
        '''
        The purpose of this function is to plot the coordination numbers 
        INPUTS:
            self.gold_coord_num:[np.array,shape=(num_gold,2)] Coordination number of each gold atom
        OUTPUTS:
            fig, ax: figure and axis of the figure
        '''
        ### IMPORTING GLOBAL VARIABLES
        from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
        from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot
        ## CREATING FIGURE
        fig, ax = create_plot()
        ## DEFINING TITLE
        ax.set_title('Coordination number of gold atoms')
        ## DEFINING X AND Y AXIS
        ax.set_xlabel('Gold coordination number', **LABELS)
        ax.set_ylabel('Number of occurances', **LABELS)  
        ## SETTING X AND Y LIMITS
        ax.set_xlim([0, 13])
        ax.set_xticks(np.arange(0, 13, 1))
        ## DEFINING THE COORDINATION NUMBER DATA
        coordination_num_data = self.gold_coord_num[:,1]
        ## FINDING ALL UNIQUE VALUES
        index, values = np.unique(coordination_num_data, return_counts=True)
        ## PLOTTING THE COORDINATION NUMBER DATA
        ax.bar(index, values, color = 'black', alpha = 1.00 )
        return fig, ax
    
    ### FUNCTION TO DETERMINE WHICH INDICES ARE FACETS VERSUS EDGES
    def find_facets_and_edges_based_on_coord_num(self, coord_num_facet_vs_edge_cutoff = 7, verbose=True):
        '''
        The purpose of this function is to distinguish between facet and edge atoms
        INPUTS:
            self: class object
            coord_num_facet_vs_edge_cutoff: [int, default = 7] coordination number to distinguish between facet and edges.
                - Values less than or equal to this number is considered an edge (or equivalently, 0)
                - Otherwise, the gold atom is considered a facet atom (or equivalently 1)
            verbose: [logical, default = True] True if you want to see outputs
        OUTPUTS:
            self.gold_surface_facet_vs_edge_classification: [np.array, shape=(num_surface_atoms, 1)] Classification of edge and surface, where:
                Values of 0 indicate edges
                Values of 1 indicate facets
            self.gold_surface_facet_index:  [np.array, shape=(num_facets, 1)]   Indices of gold surface atoms that are faceted
            self.gold_surface_edge_index:   [np.array, shape=(num_edge, 1)]     Indices of gold surface atoms that are on the edge
        '''
        ## CREATING ZERO ARRAY, ASSUMING ALL ARE EDGES
        self.gold_surface_facet_vs_edge_classification = np.zeros( self.gold_surface_num_atom )
        ## SETTING ALL VALUES THAT ARE OUTSIDE CUTOFF TO 1
        self.gold_surface_facet_vs_edge_classification[ self.gold_surface_coord_num > coord_num_facet_vs_edge_cutoff ] = 1
        ## FINDING SURFACE INDICES FOR FACETS AND EDGES
        self.gold_surface_facet_index   = np.where(self.gold_surface_facet_vs_edge_classification == 1)[0] # Returns array instead of tuple
        self.gold_surface_edge_index    = np.where(self.gold_surface_facet_vs_edge_classification == 0)[0] # Returns array instead of tuple
        
        ## PRINTING
        if verbose == True:
            print("\n----- FACETS AND EDGES OF SURFACE ATOMS -----")
            print("TOTAL SURFACE FACETS: %d"%(np.sum(self.gold_surface_facet_vs_edge_classification==1) ))
            print("TOTAL SURFACE EDGES: %d"%(np.sum(self.gold_surface_facet_vs_edge_classification==0) ))
        return
        
    ### FUNCTION TO PLOT SURFACE FACETS AND EDGES
    def plot3d_surface_facets_vs_edges(self, traj, frame = -1 ):
        '''
        The purpose of this function is to plot the surface facets compared to the edge atoms. This serves as a verification of how your gold structure is like.
        INPUTS:
            self: class object
            traj: trajectory from md.traj
            frame: [int, default = -1] frame that you want to plot
        OUTPUTS:
            xyz plot of the facets and edges
        '''
        ## IMPORTING GLOBAL VARIABLES
        from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
        from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot
        
        ## CREATING PLOT
        fig, ax = create_3d_axis_plot()
        
        ## DEFINING TITLE
        ax.set_title("Surface facet versus edge atoms")
        
        ## DEFINING LABELS
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_zlabel("z (nm)")
        
        ## SETTING DICTIONARY FOR EACH TYPE OF ATOM
        plot_dict = {
                      'facet':
                          {    'color'      : 'g',
                               's'          : 100,
                               'marker'     : 'o',
                               'label'      : 'facet atoms',
                                  },
                      'edge':
                          {    'color'      : 'r',
                               's'          : 100,
                               'marker'     : 'o',
                               'label'      : 'edge atoms',
                                  }
                        }
        
        ## DEFINING COORDINATES FOR FACETS AND EDGES
        coordinates_surface_facets  = traj.xyz[ frame, self.gold_atom_index[self.gold_surface_atom_index[self.gold_surface_facet_index]], :]
        coordinates_surface_edges   = traj.xyz[ frame, self.gold_atom_index[self.gold_surface_atom_index[self.gold_surface_edge_index]], : ]
        
        ## PLOTTING THE FACETS AND EDGES
        # FACETS
        ax.scatter( coordinates_surface_facets[:, 0], coordinates_surface_facets[:, 1], coordinates_surface_facets[:, 2], **plot_dict['facet']  )
        # EDGES
        ax.scatter( coordinates_surface_edges[:, 0], coordinates_surface_edges[:, 1], coordinates_surface_edges[:, 2], **plot_dict['edge'] )
        
        ## ADDING LEGEND
        ax.legend(loc = 'lower right')
        
        return
    
    ### FUNCTION TO FIND FACETS TO EDGES RATIO
    @staticmethod
    def calc_facets_to_edges_ratio(traj, gold_sulfur_atom_pairs, gold_atom_index, sulfur_atom_index, 
                                   gold_surface_facet_index, gold_surface_edge_index, gold_sulfur_cutoff, gold_surface_facet_vs_edge_classification, maximum_num_gold_atoms = 3, periodic = True):
        '''
        The purpose of this function is to calculate the facets to edges ratio for each sulfur atom.
            - We will do this by finding nearby sulfur-gold using a cutoff (self.gold_sulfur_cutoff)
            - This step is expected to be the slow step (due to distance calculations)
        INPUTS:
            traj: trajectory from md.traj
            gold_sulfur_atom_pairs: [np.array, shape=(num_pairs,2)] atom pairs between gold and sulfur atoms
            gold_atom_index: [np.array, shape=(num_gold_atoms, 1)] gold atom index used to generate the atom pairs
            sulfur_atom_index: [np.array, shape=(num_sulfur_atoms, 1)] sulfur atom index used to generate the atom pairs
            gold_surface_facet_index: [np.array, shape=(num_gold_surface_facets, 1)] indices of surface facet indices
            gold_surface_edge_index:   [np.array, shape=(num_edge, 1)] indices of gold surface atoms that are on the edge
            periodic: [logical, default=True] True if you want PBCs to be accounted for
            maximum_num_gold_atoms: [int, default=3] number of maximum gold atoms to search for
        OUTPUTS:
            ratio_of_facet_surface_atoms: [np.array, shape=(num_frames, num_sulfur_atoms)] ratio of facet atoms to total surrounding atoms
                - can contain nan's. If this is the case, then the sulfur atom is not nearby any of the gold atoms based on your cutoff.
                - all numpy nan's will be ignored, so no error message will be displayed
        
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
        
        # ---------- RATIO VIA CUTOFF DISTANCE ------------ #
#        ## FINDING WHEN DISTANCES ARE LESS THAN A CUTOFF
#        distances = distances < gold_sulfur_cutoff
#        
#        ## FINDING TOTAL NUMBER OF FACETS AND EDGES
#        total_edges = np.sum(distances[:, gold_surface_edge_index, :], axis = 1)
#        total_facets = np.sum(distances[:, gold_surface_facet_index, :], axis = 1)
#        ## RETURNS: num_frames, num_sulfur atoms
#        
#        ## FINDING RATIO
#        ratio_of_facet_surface_atoms = total_facets / ( total_facets +  total_edges)
#        ## NOTE: 
#        ## - This can return nan's -- occurs when you have no facets and no edges (i.e. sulfur atom is not nearby the gold atoms!)
#        ## - currently, nan's are surpressed and not use in the calculations of facets and edges ratio
        
        return ratio_of_facet_surface_atoms
    
    ### FUNCTION TO PLOT FACETS TO EDGES AS A HISTOGRAM
    def plot_hist_ratio_of_facet_surface_atoms(self, bin_width = 0.05):
        '''
        The purpose of this function is to plot the histogram ratio of facets to surface atoms
        INPUTS:
            self: class object
            bin_width: [float, default = 0.1] bin widths for the histogram
            
        OUTPUTS:
            fig, ax: figure and axis for a histogram plot between occurances and ratio of facet atoms
                NOTE:
                    - this function ignores all 'nan' arguments
        '''
        ### IMPORTING GLOBAL VARIABLES
        from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
        from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot
        ## CREATING FIGURE
        fig, ax = create_plot()
        ## DEFINING TITLE
        ax.set_title('Ratio of facet-edge surface atoms near sulfurs')
        ## DEFINING X AND Y AXIS
        ax.set_xlabel('Ratio of facet atoms to total nearby gold atoms', **LABELS)
        ax.set_ylabel('Normalized number of occurances', **LABELS)  
        ## DEFINING THE BIN
        bins = np.arange(0, 1 + bin_width, bin_width)
        ## DEFINING AXIS
        ax.set_xlim([0, 1])
        ax.set_xticks(bins)
        ## DEFINING THE DATA
        data = self.ratio_of_facet_surface_atoms
        ## REMOVING ALL NANS
        data = data[~np.isnan(data)]
        ## PLOTTING
        ax.hist( data, bins = bins, density=True, color='k')
        
        return fig, ax
        
 

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"EAM_SPHERICAL_HOLLOW" # Analysis directory
    category_dir="EAM" # category directory
    specific_dir="EAM_310.15_K_2_nmDIAM_dodecanethiol_CHARMM36_Trial_1" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
    
    '''
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"PLANAR_SIMS" # Analysis directory
    category_dir="Planar" # category directory
    specific_dir="Planar_310.15_K_dodecanethiol_10x10_CHARMM36_intffGold" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
    '''
    ### DEFINING FULL PATH TO WORKING DIRECTORY
    path2AnalysisDir=r"R:\scratch\nanoparticle_project\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir + '\\' # PC Side

    ### DEFINING FILE NAMES
    '''
    gro_file=r"sam_prod.gro" # Structural file
    xtc_file=r"sam_prod_10_ns_whole.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    '''
    gro_file=r"sam_prod_10_ns_whole_no_water_center.gro" # Structural file
    xtc_file=r"sam_prod_10_ns_whole_no_water_center.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    #%%
    from MDDescriptors.application.nanoparticle.self_assembly_structure import self_assembly_structure
    ### DEFINING INPUT DATA
    input_details = {   'traj_data'          :           traj_data,                      # Trajectory information
                         'ligand_names'      :           ['OCT', 'BUT', 'HED', 'DEC', 'DOD',],   # Name of the ligands of interest
                         'itp_file'          :           'sam.itp',                      # ITP FILE
                         'gold_atom_name'    :          'Au',                           # Atom name of gold   
                         'save_disk_space'   :          False    ,                        # Saving space
                         'gold_sulfur_cutoff':      GOLD_SULFUR_CUTOFF, # Gold sulfur cutoff
                         'split_traj'        :          25, # Number of frames to split trajectory
                         'maximum_num_gold_atoms':          3,
                         'separated_ligands' :          False,
                         'analysis_classes'  : [[ self_assembly_structure, '180814-FINAL' ], ## REUSING SELF ASSEMBLY STRUCTURE CLASS
                                                ],
                         'Pickle_loading_file': specific_dir,       ## SPECIFIC DIRECTORY FOR LOADING
                         'debug'             :          True, # Debugging
                         }
    
    ### RUNNING NANOPARTICLE GOLD STRUCTURE
    sulfur_gold_coord = nanoparticle_sulfur_gold_coordination( **input_details )

    