# -*- coding: utf-8 -*-
"""
self_assembly_structure.py
The purpose of this script is to capture the self-assembly process structural properties
Written by: Alex K. Chew (alexkchew@gmail.com, 03/22/2018)

FUNCTIONS:
    plot_convex_hull: plots the convex hull
    find_gold_gold_cutoff_by_shape: [staticmethod] Finds gold-gold cutoff based on the shape
    calc_gold_gold_distances: calculates gold-gold distance matrix
    calc_gold_gold_coord_num: calculates gold coordination numbers
    find_surface_atoms_based_on_coord_num: finds surface atoms based on coordination number
    
CLASSES:
    self_assembly_structure: calculates self-assembly structure processes

UPDATES:
    - 20180807 - inclusion of ConvexHull calculation for surface area of the gold
    - 20180814 - inclusion of coordination number calculation for self assembled structures

"""
### IMPORTING MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import mdtraj as md
from MDDescriptors.core.calc_tools import split_traj_function
import numpy as np

## CALC TOLS
import MDDescriptors.core.calc_tools as calc_tools # calc tools

### DEFINING GLOBAL VARIABLES
#GOLD_RESIDUE_NAME="AUNP" # Residue name for gold, which we will use to find self-assembly proccesses
#ATTACHED_SULFUR_ATOM_NAME="S1" # sulfur that is attached onto the gold atom
#GOLD_SULFUR_CUTOFF = 0.327 # nm, cutoff between gold and sulfur to indicate bonding

### IMPROTING GLOBAL VARIABLES
from MDDescriptors.application.nanoparticle.global_vars import LIGAND_HEAD_GROUP_ATOM_NAME as ATTACHED_SULFUR_ATOM_NAME
from MDDescriptors.application.nanoparticle.global_vars import GOLD_RESIDUE_NAME, GOLD_SULFUR_CUTOFF, GOLD_GOLD_CUTOFF_BASED_ON_SHAPE

### IMPORTING TOOLS FOR CALCULATING SASA
from MDDescriptors.geometry.accessible_hydroxyl_fraction import custom_shrake_rupley  ## CUSTOM SHRAKE AND RUPLEY BASED ON BONDI VDW RADII

### IMPORTING TOOLS TO GET THE SURFACE ATOMS
# from MDDescriptors.application.nanoparticle.nanoparticle_sulfur_gold_coordination import full_find_surface_atoms

### IMPORTING PLOTTING TOOLS
from MDDescriptors.core.plot_tools import create_3d_axis_plot_labels

## IMPORTING GENERAL TRAJ INFORMATION
from MDDescriptors.core.general_traj_info import general_traj_info

## RDF FUNCTION AND TOOLS
from MDDescriptors.application.nanoparticle.nanoparticle_rdf import plot_rdf
from MDDescriptors.geometry.rdf_extract import find_first_solvation_shell_rdf

### FUNCTION TO PLOT THE CONVEX HULL
def plot_convex_hull(hull, want_all_plots=False):
    '''
    The purpose of this function is to plot the convex hull based on a set of points
    INPUTS:
        hull: [class object] ConvexHull from scipy.spatial
        want_all_plots: [logical, default=False] True if you want the points and the convex hull in separate figures
    OUTPUTS:
        Three plots:
            - figure of the gold points by itself (shown if want_all_plots = True)
            - figure of the convex hull by itself (shown if want_all_plots = True)
            - figure of the gold points with the convex hull
    '''
    ## DEFINING POINTS
    points = hull.points
    ## PLOTTING GOLD AND HULL BY ITSELF
    if want_all_plots == True:
        ## CREATING FIGURE FOR GOLD
        fig_gold, ax_gold = create_3d_axis_plot_labels( labels = ['x (nm)', 'y (nm)', 'z (nm)'] )
        ax_gold.scatter(points[:,0], points[:,1], points[:, 2], marker = 'o', color='#FFD700', linewidth=1, edgecolors='k')
        ax_gold.set_title("GOLD ATOMS")
        
        ## CREATING FIGURE FOR THE HULL
        fig_hull, ax_hull = create_3d_axis_plot_labels( labels = ['x (nm)', 'y (nm)', 'z (nm)'] )
        ax_hull.set_title("CONVEX HULL")
        for simplex in hull.simplices:
            ax_hull.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'r-', linewidth=2)
        
    ## CREATING GENERALIZED FIGURE FOR GOLD AND HULL
    fig_gold_hull, ax_gold_hull = create_3d_axis_plot_labels( labels = ['x (nm)', 'y (nm)', 'z (nm)'] )
    ax_gold_hull.set_title("GOLD ATOMS AND CONVEX HULL")
    
    ax_gold_hull.scatter(points[:,0], points[:,1], points[:, 2], marker = 'o', color='#FFD700', linewidth=1, edgecolors='k')
    for simplex in hull.simplices:
        ax_gold_hull.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'r-', linewidth=2)
    return

### FUNCTION TO CALCULATE LIGAND GRAFTING DENSITIES BASED ON NUMBER OF LIGANDS AND SURFACE AREA
def calc_ligand_grafting_density(num_ligands, surface_area ):
    '''
    The purpose of this function is to calculate ligand grafting density based on surface area and number of ligands
    INPUTS:
        num_ligands: [int] number of ligands grafted onto the surface
        surface_area: [float] surface area that the ligands are grafted onto in nm^2
    OUTPUTS:
        area_angs_squared_per_ligand: [float] area (angstroms^2) per ligand
        ligand_per_area_nm_squared: [float] ligand per area (nm^2)
    '''
    ## FINDING GRAFTING DENSITY IN TERMS OF AREA SQUARED PER LIGAND
    area_angs_squared_per_ligand = surface_area / float(num_ligands) * 100 # in Angstroms^2 per ligand
    ## FINDING GRAFTING DENSITY IN TERMS OF LIGAND PER NM2
    ligand_per_area_nm_squared = num_ligands / surface_area
    return area_angs_squared_per_ligand, ligand_per_area_nm_squared
    

### FUNCTION TO FIND GOLD GOLD CUTOFF BASED ON SHAPE
def find_gold_gold_cutoff_by_shape( gold_shape ):
    '''
    The purpose of this function is to find the gold gold cutoff based on predefined shape calculations
    INPUTS:
        gold_shape: [str] shape of the gold
    OUTPUTS:
        gold_gold_cutoff: [float] cutoff for gold-gold nearest neighbors calculation
    '''
    if gold_shape in GOLD_GOLD_CUTOFF_BASED_ON_SHAPE.keys():
        gold_gold_cutoff = GOLD_GOLD_CUTOFF_BASED_ON_SHAPE[gold_shape]
    else:
        print("ERROR! Shape could not be found: %s"%(gold_shape))
        print("If this is a new shape, you will need to update the global variable, 'GOLD_GOLD_CUTOFF_BASED_ON_SHAPE'")
        print("The gold-gold cutoff is defined as when the gold-gold RDF reaches a minimum after the first solvation peak")
    return gold_gold_cutoff

### FUNCTION TO FIND COORDINATION NUMBER OF THE GOLD ATOMS
def calc_coord_num(distance_matrix, cutoff ):
    '''
    The purpose of this function is to find the coordination number of each gold atom or metal atom.
    INPUTS:
        distance_matrix: [np.array, shape=(num_atom,num_atom)] distance matrix of gold-gold, e.g.
        cutoff: [float] cutoff for gold-gold pairs
    OUTPUTS:
        coord_num:  numpy stacked version of: 
            index: [np.array, shape=(Nx1)] atom index of the gold atoms
            coordination_number: [np.array, shape=(Nx1)] number of gold atoms coordinated for the index
            e.g.
            [[0, 5], [1,7]] <-- atom index 0 has 5 nearest neighbors, and so on
    '''
    ## DEFINING A CRITERIA TO FIND COORDINATION NUMBERS
    match_criteria = np.where( (distance_matrix < cutoff) & (distance_matrix > 0))
    # RETURNS: match_criteria is a tuple of size two: 
    #   First tuple contains unique indexes of the atoms
    #   Second tuple contains number of occurances
    ## FINDING INDEX AND COORDINATION NUMBER: Simply count the unique number of times you find a bond
    index, coordination_number = np.unique(match_criteria, return_counts=True)
    
    ## CONCATENATING RESULTS
    coord_num = np.column_stack( (index, coordination_number) )
    
    return coord_num

### FUNCTION TO DETERMINE INDICES OF GOLD SURFACE ATOMS
def find_surface_atoms_based_on_coord_num(gold_coord_num, gold_atom_index, coord_num_surface_to_bulk_cutoff = 11, verbose=True):
    '''
    The purpose of this function is to find the surface atoms based on coordination number
    We will based the surface atoms as those that are less than or equal to the coord number cutoff. Bulk atoms are those that are > that coord_num_cutoff
    INPUTS:
        gold_coord_num: [np.array, shape=(num_gold, 2)] indices and coordination number of the gold atoms
        gold_atom_index: [np.array, shape=(num_gold,1)] atom index of the gold based on your trajectory
        coord_num_surface_to_bulk_cutoff: [int, default = 11] coordination number that defines the surface atoms. Values less than or equal to this value is considered "surface atoms"
        verbose: [logical, default = False] True if you want to print the output of the surface identification function
    OUTPUTS:
        gold_surface_indices: [np.array] array with the indices of the surface atoms
        gold_surface_atom_index: [np.array] array with gold-surface atom indices (referenced to traj.topology)
        gold_surface_coord_num: [np.array, shape=(num_surface_atoms)] surface coordination numbers
        gold_surface_num_atom: [int] total number of surface atoms
    '''
    ## FINDING TOTAL GOLD ATOMS
    gold_num_atoms = len(gold_atom_index)
    ## FINDING LOCATIONS WHERE THE CUTOFF IS LESS THAN OR EQUAL TO CUTOFF
    gold_surface_indices = gold_coord_num[gold_coord_num[:,1]<=coord_num_surface_to_bulk_cutoff,0]
    ## FINDING SURFACE ATOM INDEX
    gold_surface_atom_index = gold_atom_index[gold_surface_indices]
    ## FINDING ALL COORDINATION NUMBERS OF THE SURFACE ATOMS
    gold_surface_coord_num = gold_coord_num[gold_surface_indices, 1]
    ## FINDING TOTAL GOLD SURFACE ATOMS
    gold_surface_num_atom  = len(gold_surface_indices)
    ## PRINTING
    if verbose == True:
        print("\n----- SURFACE AND BULK ATOMS -----")
        print("TOTAL ATOMS: %d"%(gold_num_atoms) )
        print("TOTAL SURFACE ATOMS: %d"%(gold_surface_num_atom))
        print("TOTAL BULK ATOMS: %d"%( gold_num_atoms - gold_surface_num_atom  ))
    
    return gold_surface_indices, gold_surface_atom_index, gold_surface_coord_num, gold_surface_num_atom

### FUNCTION TO FIND SURFACE ATOMS USING PREVIOUSLY DEFINED FUNCTIONS
def full_find_surface_atoms( traj, cutoff, gold_atom_index, coord_num_surface_to_bulk_cutoff = 11, frame = -1 ,verbose = False, periodic = True ):
    '''
    The purpose of this function is to find the surface atoms of gold atoms
    INPUTS:
        traj: trajectory from md.traj
        cutoff: [float] gold gold cutoff, used to define coordination
        gold_atom_index: [np.array, shape=(num_atoms, 1)] atom indices that you want to develop a pair distance matrix for
        coord_num_surface_to_bulk_cutoff: [int, default = 11] coordination number that defines the surface atoms. Values less than or equal to this value is considered "surface atoms"
        frame: [int, default=-1] frame to calculate gold-gold distances
        verbose: [logical, default = False] True if you want to print the output of the surface identification function
        periodic: [logical, default=True] True if you want PBCs to be accounted for
    OUTPUTS:
        gold_gold_coordination: [dict] dictionary containing the following:
            gold_coord_num: [np.array, shape=(num_gold, 2)] indices and coordination number of the gold atoms
            gold_gold_cutoff: [float] cutoff for gold-gold nearest neighbors calculation
            gold_surface_indices: [np.array] array with the indices of the surface atoms
            gold_surface_atom_index: [np.array] array with gold-surface atom indices (referenced to traj.topology)
            gold_surface_coord_num: [np.array, shape=(num_surface_atoms)] surface coordination numbers
            gold_surface_num_atom: [int] total number of surface atoms
        
    EXTERNAL FUNCTIONS USED:
        find_gold_gold_cutoff_by_shape: finds gold-gold cutoff based on the shape of the gold atoms
        calc_pair_distances_with_self_single_frame: calculates pair distances for a single frame
        calc_coord_num: calculates corodination numbers based on the distance matrix
        find_surface_atoms_based_on_coord_num: finds surface atoms based on the coordination numbers
    '''
    ## FINDING GOLD CUTOFF BASED ON SHAPE
    # gold_gold_cutoff = find_gold_gold_cutoff_by_shape( gold_shape = gold_shape )
    ## CALCULATING DISTANCE MATRIX
    distances_matrix = calc_tools.calc_pair_distances_with_self_single_frame( traj = traj,
                                                                               atom_index = gold_atom_index,
                                                                               frame = frame,
                                                                               periodic = periodic,
                                                                              )
    ## CALCULATING COORDINATION NUMBER
    gold_coord_num = calc_coord_num( distance_matrix = distances_matrix,
                                     cutoff = cutoff,
                                     )
    ## FINDING SURFACE ATOMS
    gold_surface_indices, gold_surface_atom_index, gold_surface_coord_num, gold_surface_num_atom = \
                    find_surface_atoms_based_on_coord_num( gold_coord_num = gold_coord_num,
                                                           gold_atom_index = gold_atom_index,
                                                           coord_num_surface_to_bulk_cutoff = coord_num_surface_to_bulk_cutoff,
                                                           )
    ## CREATING DICTIONARY TO CAPTURE THE VARIABLES
    gold_gold_coordination = {      
                                'gold_coord_num'            : gold_coord_num,           # Gold coordination numbers in a form of a list
                                'gold_gold_cutoff'          : cutoff,         # Cutoff as a floating point number
                                'gold_surface_indices'      : gold_surface_indices,     # Surface indices
                                'gold_surface_atom_index'   : gold_surface_atom_index,  # Surface atom index
                                'gold_surface_coord_num'    : gold_surface_coord_num,   # Coordination number of gold atoms on the surface
                                'gold_surface_num_atom'     : gold_surface_num_atom,    # Total number of atoms on the surface
                              }
                    
    return gold_gold_coordination

### FUNCTION TO DETERMINE WHICH INDICES ARE FACETS VERSUS EDGES
def find_facets_and_edges_based_on_coord_num(gold_gold_coordination, coord_num_facet_vs_edge_cutoff = 7, verbose=True):
    '''
    The purpose of this function is to distinguish between facet and edge atoms
    INPUTS:
        self: class object
        coord_num_facet_vs_edge_cutoff: [int, default = 7] coordination number to distinguish between facet and edges.
            - Values less than or equal to this number is considered an edge (or equivalently, 0)
            - Otherwise, the gold atom is considered a facet atom (or equivalently 1)
        verbose: [logical, default = True] True if you want to see outputs
    OUTPUTS:
        gold_gold_surface_facet_edge_dict: [dict] containing all information shown below:
            gold_surface_facet_vs_edge_classification: [np.array, shape=(num_surface_atoms, 1)] Classification of edge and surface, where:
                Values of 0 indicate edges
                Values of 1 indicate facets
            gold_surface_facet_index:  [np.array, shape=(num_facets, 1)]   Indices of gold surface atoms that are faceted
            gold_surface_edge_index:   [np.array, shape=(num_edge, 1)]     Indices of gold surface atoms that are on the edge
    '''
    ## CREATING ZERO ARRAY, ASSUMING ALL ARE EDGES
    gold_surface_facet_vs_edge_classification = np.zeros( gold_gold_coordination['gold_surface_num_atom'] )
    ## SETTING ALL VALUES THAT ARE OUTSIDE CUTOFF TO 1
    gold_surface_facet_vs_edge_classification[ gold_gold_coordination['gold_surface_coord_num'] > coord_num_facet_vs_edge_cutoff ] = 1
    ## FINDING SURFACE INDICES FOR FACETS AND EDGES
    gold_surface_facet_index   = np.where(gold_surface_facet_vs_edge_classification == 1)[0] # Returns array instead of tuple
    gold_surface_edge_index    = np.where(gold_surface_facet_vs_edge_classification == 0)[0] # Returns array instead of tuple
    
    ## PRINTING
    if verbose == True:
        print("\n----- FACETS AND EDGES OF SURFACE ATOMS -----")
        print("TOTAL SURFACE FACETS: %d"%(np.sum(gold_surface_facet_vs_edge_classification==1) ))
        print("TOTAL SURFACE EDGES: %d"%(np.sum(gold_surface_facet_vs_edge_classification==0) ))
        
    ## CREATING DICTIONARY TO STORE ALL SURFACE INFORMATION
    gold_gold_surface_facet_edge_dict = {
                                        'gold_surface_facet_vs_edge_classification' : gold_surface_facet_vs_edge_classification,
                                        'gold_surface_facet_index'                  : gold_surface_facet_index,
                                        'gold_surface_edge_index'                   : gold_surface_edge_index,            
                                        }
        
    return gold_gold_surface_facet_edge_dict


####################################################################################
### CLASS FUNCTION TO CALCULATE STRUCTURAL PROPERTIES OF SELF-ASSEMBLY PROCESSES ###
####################################################################################
class self_assembly_structure:
    '''
    The purpose of this class it to take the trajectory data of a self-assembly process and characterize it. 
    We would like to know, for example, how many ligands get attached over time
    INPUTS:
        traj_data: Data taken from import_traj class
        gold_residue_name: [str] 
            residue name of the gold atoms
        sulfur_atom_name: [str] 
            name of the sulfur atoms
            If None, it will look for all sulfur atoms
        ps_per_frame: [str] 
            picoseconds per frame of your simulation
        gold_sulfur_cutoff: [float] 
            gold sulfur cutoff for bonding
        gold_shape: [str, default=EAM] 
            shape of the gold: <-- used to get a gold-gold cutoff for coordination number, e.g.
                EAM: embedded atom model
                spherical: based on fcc {111} cut
                hollow: based on perfectly uniform sphere that has a hollow core
        NOTE: Alternatively, you can use the cutoff given from the RDF coordination number. ** not currently implemented, but tested and found relatively good correlations **
        coord_num_facet_vs_edge_cutoff: [int, default=7]
            coordination number between facet and edge cutoffs
        coord_num_surface_to_bulk_cutoff: [int, default = 11] 
            coordination number that defines the surface atoms. Values less than or equal to this value is considered "surface atoms"
        split_traj: [int] 
            splitting trajectory frames for issues with memory error
        gold_optimize: [logical] 
            True if you want to optimize gold indices by removing those within the core 
        save_disk_space: [logical, default = True] 
            True if you want to save disk space and remove variables:
    OUTPUTS:
        ## INPUTS
            self.gold_residue_name: gold residue name (string)
            self.sulfur_atom_name: sulfur atom name (string)
            self.ps_per_frame: picoseconds per frame (integer)
            self.gold_sulfur_cutoff: gold sulfur cutoff (float)
        ## TRAJECTORY DETAILS
            self.total_frames: Total frames of trajectory (integer)
            self.frames_ps: frame vector in picoseconds
            self.frames_ns: frame vector in nanoseconds
        ## SULFUR AND GOLD INDEX
            self.sulfur_index: index of all the sulfurs
            self.total_sulfur: total sulfur atoms
            self.gold_index: index of all the gold atoms
            self.total_gold: total gold atoms
        ## GOLD ATOM PAIRS
            self.gold_gold_atom_pairs: [np.array, shape=(num_pairs, 2)] atom pairs between gold-gold atoms
        ## SULFUR AND GOLD BONDING
            self.num_gold_sulfur_bonding_per_frame: total number of gold sulfur unique bonding per frame
        ## COORDINATION NUMBERS OF GOLD
            ## GOLD SURFACE INFORMATION
                gold_gold_coordination: [dict] dictionary containing information about gold-gold coordination, e.g.
                    gold_coord_num: [np.array, shape=(num_gold, 2)] indices and coordination number of the gold atoms
                    gold_gold_cutoff: [float] cutoff for gold-gold nearest neighbors calculation
                    gold_surface_indices: [np.array] array with the indices of the surface atoms
                    gold_surface_atom_index: [np.array] array with gold-surface atom indices (referenced to traj.topology)
                    gold_surface_coord_num: [np.array, shape=(num_surface_atoms)] surface coordination numbers
                    gold_surface_num_atom: [int] total number of surface atoms
            ## FACETS AND EDGES
                gold_gold_surface_facet_edge_dict: [dict] dictionary containing information about facet vs. edges
                    gold_surface_facet_vs_edge_classification: [np.array, shape=(num_surface_atoms, 1)] Classification of edge and surface, where:
                        Values of 0 indicate edges
                        Values of 1 indicate facets
                    gold_surface_facet_index:  [np.array, shape=(num_facets, 1)]   Indices of gold surface atoms that are faceted
                    gold_surface_edge_index:   [np.array, shape=(num_edge, 1)]     Indices of gold surface atoms that are on the edge
        ## CONVEX HULL
            self.hull: [class object] hull object from ConvexHull
        ## SURFACE AREA
            self.surface_area_spherical: [float] surface area based on diameter and assuming spherical geometry
            self.surface_area_hull: [float] surface area based on ConvexHull in nm2
        ## LIGAND DENSITY DETAILS
            ## ASSUMING SPHERICAL
            self.area_angs_per_ligand_spherical: [float] area per ligand (angstrom^2/lig)
            self.ligand_per_area_spherical: [float] ligand per area (ligand/nm^2)
            ## ASSUMING CONVEX HULL
            self.area_angs_per_ligand_hull: [float] area per ligand (angstrom^2/lig)
            self.ligand_per_area_hull: [float] ligand per area (ligand/nm^2)
    FUNCTIONS:
        find_sulfur_gold_index: Finds all sulfur and gold indexes
        optimize_gold_index: optimizes gold index based on the cutoff: 
            diameter - 6 * cutoff between sulfur + gold
            center of gold will be taken as the mean of all the gold positions
        ## RDF FUNCTIONS
            calc_rdf_gold_gold_single_frame: calculates gold-gold RDF for a single frame
            calc_rdf_first_solvation_shell: calculates the first solvation shell for gold-gold RDF
        ## GOLD COORDINATION NUMBER FUNCTIONS
            find_gold_gold_coordination_number: calculates gold gold coordination number
        ## GOLD SULFUR BONDING
            find_sulfur_gold_bonding: Finds all sulfur gold bonding
            find_ligand_density: Finds ligand density and calculates gold diameter
        ## AREAS
            calc_gold_sasa: [DEPRECIATED] calculates the gold sasa as an estimation of gold surface area
            calc_gold_area_by_ConvexHull: calculates gold surface area based on Convex Hull
        ## DEBUGGING FUNCTIONS
            plot_optimized_gold_index: plots gold index within a radius for optimized gold
        ## ACTIVE FUNCTIONS
            plot_ConvexHull: plots the convex hull
            plot_gold_gold_rdf: plots gold-gold RDF
        
    ALGORITHM:
        - Find sulfur and gold atom index
        - Find distances between gold and all sulfur atoms
        - Find ligand density
    '''
    ## INITIALIZING
    def __init__(self, 
                 traj_data,                                     # Trajectory data
                 gold_residue_name=GOLD_RESIDUE_NAME,           # Residue name of gold
                 sulfur_atom_name = ATTACHED_SULFUR_ATOM_NAME,  # Atom name of sulfur
                 gold_shape = 'spherical',                      # Shape of the gold atom
                 coord_num_facet_vs_edge_cutoff = 7,            # Coordination number for facet vs. edge cutoff    
                 coord_num_surface_to_bulk_cutoff = 11,         # coordination number for surface to bulkcutoff
                 ps_per_frame = 50,                             # Number of picoseconds per frame (default: 50)
                 gold_sulfur_cutoff = GOLD_SULFUR_CUTOFF,       # Gold sulfur cutoff
                 split_traj = 50,                               # Split trajectory frequency (50 frames splitted to run this calculation, lower if job fails!)         
                 gold_optimize = True,                          # True if you want to optimize the gold atoms by omission
                 save_disk_space = True,                        # True if you want to save disk space by removing unnecessary variables
                                 ):
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__))
        
        ## STORING INPUTS
        self.gold_residue_name  =  gold_residue_name
        self.sulfur_atom_name = sulfur_atom_name
        self.ps_per_frame       =  ps_per_frame
        self.gold_shape = gold_shape
        self.coord_num_facet_vs_edge_cutoff = coord_num_facet_vs_edge_cutoff
        self.coord_num_surface_to_bulk_cutoff = coord_num_surface_to_bulk_cutoff
        self.gold_sulfur_cutoff =   gold_sulfur_cutoff
        self.gold_optimize = gold_optimize
        
        ## DEFINING TRAJECTORY
        traj = traj_data.traj
        
        ## GETTING TRAJECTORY DETAILS
        self.traj_info = general_traj_info(traj_data = traj_data)
        self.total_frames = self.traj_info.total_frames
        ## FINDING TIME THAT CORRESPONDS TO THE NUMBER OF BONDING PER FRAME
        self.frames_ps=np.arange(0, self.total_frames) * self.ps_per_frame
        self.frames_ns=self.frames_ps/1000.0
                
        ## FINDING ALL GOLD AND SULFUR INDEXES
        self.find_sulfur_gold_index(traj)
        
        ## FINDING GOLD PAIRS
        self.gold_gold_atom_pairs = calc_tools.create_atom_pairs_with_self(self.gold_index)[0]
        
        ## FINDING OPTIMAL GOLD INDICES
        self.optimize_gold_index(traj)
        
        ## FINDING ALL SULFUR GOLD BONDING
        # NOTE: SPLITTING TO ENSURE THAT THIS FUNCTION WORKS
        sulfur_gold_index_input = {
                                    'sulfur_index'      : self.sulfur_index, # Sulfur index
                                    'gold_index'        : self.gold_optimize_index, # Gold index
                                    'gold_sulfur_cutoff': self.gold_sulfur_cutoff, # Cutoff for gold and sulfur
#                                    'split_traj'        : split_traj, # Frequency of splitting trajectory
                                    'return_sulfur_index_array': True,
                                    }
        
        self.num_gold_sulfur_bonding_per_frame, self.num_gold_sulfur_bonding_indices = self.find_sulfur_gold_bonding(traj = traj,
                                                                                                                     **sulfur_gold_index_input)
        
        '''
        
        self.num_gold_sulfur_bonding_per_frame, self.num_gold_sulfur_bonding_indices = split_traj_function(traj = traj, 
                                                                                                           input_function = self.find_sulfur_gold_bonding,
                                                                                                           **sulfur_gold_index_input
                                                                                                           )
        '''
        
        #####################################
        ### RADIAL DISTRIBUTION FUNCTIONS ###
        #####################################
        ## CALCULATING THE RDF
        self.calc_rdf_gold_gold_single_frame( traj = traj )
        ## FINDING THE FIRST SOLVATION SHELL
        self.calc_rdf_first_solvation_shell()
        
        ############################
        ### COORDINATION NUMBERS ###
        ############################
        ## FINDING GOLD CUTOFF BASED ON SHAPE
        gold_gold_cutoff = find_gold_gold_cutoff_by_shape( gold_shape = gold_shape )
        ## DEFINING COORDINATION NUMBER BASED ON THIS CUTOFF
        self.gold_gold_coordination, self.gold_gold_surface_facet_edge_dict =  self.find_gold_gold_coordination_number(traj = traj, cutoff = gold_gold_cutoff)
        ## DEFINING COORDINATION NUMBER BASED ON RDF
        # self.gold_gold_coordination_based_on_rdf, self.gold_gold_surface_facet_edge_dict_based_on_rdf =  self.find_gold_gold_coordination_number(traj = traj, cutoff = self.gold_gold_rdf_first_solv_shell['min']['r'])
        
        #################################
        ### SURFACE AREA CALCULATIONS ###
        #################################
        ## FINDING SURFACE AREA BASED ON CONVEX HULL
        self.calc_gold_area_by_ConvexHull(traj = traj)
        ## APPROXIMATING SURFACE AREA BASED ON SPHERICAL
        self.surface_area_spherical = 4 * np.pi * (self.gold_diameter/2.0)**2
        
        ## FINDING LIGAND DENSITY
        self.find_ligand_density()            
        
        ## RUNNING SASA
        # self.calc_gold_sasa(traj=traj)
        
        ## PRINTING SUMMARY
        self.print_summary()
        ## CLEANING DISK
        self.clean_disk()
        
        ## CLEANING VARIABLES
    ### FUNCTION TO CLEAN DISK
    def clean_disk(self, save_disk_space = True):
        '''
        The purpose of this function is to clean the memory of variables
        '''
        if save_disk_space == True:
            self.gold_atom_pairs = []
        
    ### FUNCTION TO PRINT SUMMARY
    def print_summary(self):
        ''' This function simply prints a summary '''
        print("\n----- SUMMARY -----")
        print("self_assembly ligand structures were computed!")
        print("Total sulfurs: %s"%(self.total_sulfur))
        print("Total gold: %s"%(self.total_gold))
        print("Total sulfur bound at the end: %s"%(self.num_gold_sulfur_bonding_per_frame[-1]))
        print("Gold diameter: %.2f"%(self.gold_diameter))
        print("---- LIGAND GRAFTING DENSITIES ----")
        print("Total ligands per nm2 (assuming spherical): %.2f"%(self.ligand_per_area_spherical) )
        print("Total ligands per nm2 (assuming convex hull): %.2f"%(self.ligand_per_area_hull) )
        print("Total area (Angstroms^2) per ligand (assuming spherical): %.2f"%(self.area_angs_per_ligand_spherical))
        print("Total area (Angstroms^2) per ligand (assuming convex hull): %.2f"%(self.area_angs_per_ligand_hull))
        
    ### FUNCTION TO LIMIT THE NUMBER OF GOLD INDEX
    def optimize_gold_index(self, traj):
        '''
        The purpose of this script is to optimize gold index, if necessary. The main idea here is that the gold atoms do not move. 
        Thus, if we focus on the surface gold atoms, we can quickly find gold to sulfur distances
        IMPORTANT NOTE: This script takes the first trajectory as its way of finding gold index
        INPUTS:
            self: class property
        OUTPUTS:
            self.gold_diameter: diameter in nm
        '''
        ## FINDING APPROXIMATE DIAMETER
        initial_traj=traj[0]
        ## CREATING ATOM PAIRS BETWEEN GOLD-GOLD
        # gold_gold_atom_pairs = np.array([ [gold_1_index,gold_2_index] for gold_1_index in self.gold_index for gold_2_index in self.gold_index 
                                                  # if gold_1_index != gold_2_index])
        ## FINDING DISTANCES BETWEEN GOLD ATOMS BY CREATING ATOM PAIRS
        gold_gold_distances = md.compute_distances( 
                                            traj        = initial_traj,             # Trajectory
                                            atom_pairs  = self.gold_gold_atom_pairs,  # Atom pairs
                                            periodic    = True,             # Periodic boundary (always true)                                                                                          
                                         )[0] # RETURNS ( NUM_SULFUR X NUM_GOLD ) numpy array
        ## FINDING APPROXIMATE DIAMETER
        self.gold_diameter = np.max(gold_gold_distances) # in nms
        
        ## PRINTING
        print("Gold optimization is: %s"%(self.gold_optimize))
        if self.gold_optimize is True:
            
            ## DEFINING POSITIONS
            self.gold_positions = initial_traj.xyz[:,self.gold_index,:][0]
            
            ## FINDING APPROXIMATE CENTER
            self.center = np.mean(self.gold_positions, axis = 0)
            
            ## FINDING DISTANCES BETWEEN GOLD ATOMS AND THE CENTER
            self.gold_distance_from_center = np.sqrt(  np.sum( (self.gold_positions - self.center)**2, axis=1 ) )
            
            ## DEFINING A CUTOFF (twice the cutoff, just in case!)
            self.gold_optimize_cutoff = self.gold_diameter/2.0 - 2 * self.gold_sulfur_cutoff
            
            ## FINDING WHEN THIS IS TRUE
            self.gold_optimize_index = np.where(self.gold_distance_from_center > self.gold_optimize_cutoff)
            ## NOW, GETTING NEW GOLD INDEX
            self.gold_optimize_index = [ self.gold_index[each_index] for each_index in self.gold_optimize_index[0] ]
            ## PRINTING
            print("SHORTING GOLD INDEX LIST FROM %d to %d..."%(len(self.gold_index), len(self.gold_optimize_index)))
        else:
            self.gold_optimize_index = self.gold_index
            print("Not changing the gold index! (leave alone!)")
        return
    
    ### FUNCTION TO PLOT THE GOLDS
    def plot_optimized_gold_index(self):
        '''
        The purpose of this function is to plot the gold index
        INPUTS:
            self: class property
        OUTPUTS:
            
        '''
        ## FUNCTION TO DRAWING SPHERE
        def draw_sphere(ax, radius, center, color='r'):
            '''
            This function simply draws a sphere given a radius and center
            INPUTS:
                ax: axis of your figure
                radius: [float] radius of sphere
                center: [3 x 1 numpy] coordinates of the center
                color: [string] color of sphere
            OUTPUTS:
                ax: updated axis
            '''
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = radius*np.cos(u)*np.sin(v) + center[0]
            y = radius*np.sin(u)*np.sin(v) + center[1]
            z = radius*np.cos(v) + center[2]
            ax.plot_wireframe(x, y, z, color=color)
            return ax
        
        ## MAIN SCRIPT FOR THIS FUNCTION
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D # For 3D axes
        ## CREATING FIGURE
        fig = plt.figure(); ax = fig.add_subplot(111, projection='3d', aspect='equal')
        
        # SETTING X, Y, Z labels
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel('z (nm)')
    
        ## PLOTTING
        ax.scatter(self.gold_positions[:,0], self.gold_positions[:,1], self.gold_positions[:,2],color='yellow', alpha = 0.4, s=100, linewidth=1.5)
        
        ## PLOTTING CENTER
        ax.scatter(self.center[0], self.center[1], self.center[2], color='black', alpha = 0.8, s=200, linewidth=1.5 )
        
        ## PLOTTING SPHERES
        ax = draw_sphere(ax, radius = self.gold_diameter/2.0, center=self.center , color='r')

        ## PLOTTING CUTOFF
        ax = draw_sphere(ax, radius = self.gold_optimize_cutoff, center=self.center , color='b')

        return
        
    ### FUNCTION TO FIND DISTANCES BETWEEN SULFUR AND GOLD
    @staticmethod
    def find_sulfur_gold_bonding(traj, sulfur_index, gold_index, gold_sulfur_cutoff,
                                 return_sulfur_index_array = False):
        '''
        The purpose of this function is to find the sulfur-gold bonding
        INPUTS:
            traj: trajectory from md.traj
            sulfur_index: (list) index of the sulfur atoms
            gold_index: (list) index of the gold atoms
            gold_sulfur_cutoff: (float) cutoff between gold and sulfur to be considered bonded
            return_sulfur_index_array: [logical]
                True if you want the sulfur indices
        OUTPUTS:
            num_gold_sulfur_bonding_per_frame: number of gold sulfur bonding per  frame
        '''
        ## FINDING TOTAL FRAMES
        total_frames = len(traj)
        total_sulfur = len(sulfur_index)
        total_gold = len(gold_index)
 
        ## GENERATING ATOM PAIRS
        sulfur_gold_atom_pairs = calc_tools.create_atom_pairs_list(gold_index, sulfur_index)
        # np.array([ [each_sulfur_index,each_gold_index] for each_sulfur_index in sulfur_index for each_gold_index in gold_index])
        # RETURNS SULFUR GOLD PAIRS LIKE: [[S1, Au_1], [S1, Au_2], etc.]
        

        ## CALCULATING DISTANCES
        distances = md.compute_distances( 
                                            traj        = traj,             # Trajectory
                                            atom_pairs  = sulfur_gold_atom_pairs,  # Atom pairs
                                            periodic    = True,             # Periodic boundary (always true)                                                                                          
                                         ) # RETURNS TIMEFRAME X ( NUM_SULFUR X NUM_GOLD ) numpy array
        
        ## RESHAPING THE DISTANCES
        reshape_distances = distances.reshape(total_frames, total_sulfur, total_gold)
        # RETURNS TIMEFRAME X NUM SULFUR X NUM GOLD (3D matrix)
        
        ## FINDING DISTANCES THAT ARE BELOW A RANGE
        logical_matrix = np.any(reshape_distances < gold_sulfur_cutoff,axis=2)
        num_gold_sulfur_bonding_per_frame =np.sum(logical_matrix,axis=1)
        # RETURNS total sulfur bound onto the surface over time as a list

        if return_sulfur_index_array is True:
            return num_gold_sulfur_bonding_per_frame, logical_matrix
        return num_gold_sulfur_bonding_per_frame
        
    ### FUNCTION TO FIND ALL SULFUR AND GOLD INDEXES
    def find_sulfur_gold_index(self, traj):
        '''
        The purpose of this function is to find the sulfur and gold index
        INPUTS:
            self: class property
            traj: trajectory from md.traj
        OUTPUTS:
            self.sulfur_index: [np.array, shape=(num_sulfur, 1)] index of all the sulfurs
            self.total_sulfur: [int] total sulfur atoms
            self.gold_index: [np.array, shape=(num_gold, 1)] index of all the gold atoms
            self.total_gold: [int] total gold atoms
        '''
        ## FINDING SULFUR INDEX
        if self.sulfur_atom_name is None:
            print("Since sulfur name is None, we are expanding sulfur indices to all possible sulfurs")
            self.sulfur_index = np.array([atoms.index for atoms in traj.topology.atoms if atoms.element.symbol == "S" ])
        else:            
            self.sulfur_index = np.array([atoms.index for atoms in traj.topology.atoms if atoms.name== self.sulfur_atom_name ])
        ## FINDING TOTAL SULFUR
        self.total_sulfur = len(self.sulfur_index)
        
        ## FINDING GOLD INDEX
        self.gold_index  = np.array([ atoms.index for atoms in  traj.topology.atoms if atoms.residue.name == self.gold_residue_name])
        self.total_gold  = len(self.gold_index)
        return
    
    ### FINDING FINAL AREA PER LIGAND / LIGAND PER NM^2
    def find_ligand_density(self):
        '''
        The purpose of this function is to find the gold-gold area, then find the ligand density.
        NOTE: the diameter is found by the maximum distance of gold atoms
        INPUTS:
            self: class property
        OUTPUTS:
            ## ASSUMING SPHERICAL
            self.area_angs_per_ligand_spherical: [float] area per ligand (angstrom^2/lig)
            self.ligand_per_area_spherical: [float] ligand per area (ligand/nm^2)
            ## ASSUMING CONVEX HULL
            self.area_angs_per_ligand_hull: [float] area per ligand (angstrom^2/lig)
            self.ligand_per_area_hull: [float] ligand per area (ligand/nm^2)
        '''            
        ## DEFINING NUMBER OF LIGANDS GRAFTED ON
        num_ligands = self.num_gold_sulfur_bonding_per_frame[-1]
        ## FINDING FINAL NUMBER OF SULFURS
        self.area_angs_per_ligand_spherical,  self.ligand_per_area_spherical = calc_ligand_grafting_density(num_ligands = num_ligands,
                                                                                                            surface_area = self.surface_area_spherical)
        ## FINDING FINAL NUMBER OF SULFURS
        self.area_angs_per_ligand_hull,  self.ligand_per_area_hull = calc_ligand_grafting_density(num_ligands = num_ligands,
                                                                                                  surface_area = self.surface_area_hull)
        
        return
    
    ### FUNCTION TO FIND THE SASA OF THE GOLD
    def calc_gold_sasa(self, traj, frame_index = -1, probe_radius = 0.01, n_sphere_points = 3840 ): # 0.14 960
        ''' **DEPRECIATED
        The purpose of this function is to calculate the gold sasa value to get an estimate of the surface area. This works well if you have a gold core that is well spaced. Note that gold has a vdw radii of 0.213 nm
        INPUTS:
            self: class property
            traj: traj from md.traj
            frame_index: [int, default=-1] frame that you want the sasa for
            probe_radius: [float, default=0.14] probe radius in nm
            n_sphere_points: [int, default=960] number of spherical points to define a sphere
        OUTPUTS:
            
        ALGORITHM:
            - trim the trajectory to be the last frame
            - trim the system to include only gold atom indices
            - calculate the sasa
            - find the total sasa of the atoms -- potentially, we can just choose the sasa of the surface atoms
        '''
        ## TRIMMING THE TRAJECTORY
        current_traj = traj[frame_index]
        
        ## FINDING ALL SURFACE ATOMS
        gold_coord_num, gold_gold_cutoff, gold_surface_indices, \
        gold_surface_atom_index, gold_surface_coord_num, gold_surface_num_atom = full_find_surface_atoms( traj = current_traj,
                                                                                                          gold_atom_index = np.array(self.gold_index),
                                                                                                          gold_shape = 'spherical',
                                                                                                          verbose = True
                                                                                                         )
        
        self.gold_surface_atom_index = gold_surface_atom_index
        
        ## USING ATOM SLICE
        self.slice_traj = current_traj.atom_slice(atom_indices = self.gold_index, inplace=False )
        
        
        # RETURNS TRAJECTORY WITH JUST THE GOLD INDICES
        ## CALCULATING SASA
        self.sasa = custom_shrake_rupley(traj=self.slice_traj[:],
                     probe_radius=probe_radius, # in nms
                     n_sphere_points=n_sphere_points, # Larger, the more accurate
                     mode='atom' # Extracted areas are per atom basis
                     ) ## OUTPUT IN NUMBER OF FRAMES X NUMBER OF FEATURES, e.g. shape = 1, 465
        
        ## FINDING THE SURFACE AREA VIA SASA
        self.gold_surface_area_based_on_sasa = np.sum(self.sasa[:, self.gold_surface_atom_index])
        self.ligand_per_sasa_area_nm = self.num_gold_sulfur_bonding_per_frame[-1] / self.gold_surface_area_based_on_sasa
        
        return
    
    ### FUNCTION TO CALCULATE THE CONVEX HULL AND THE SURFACE AREA
    def calc_gold_area_by_ConvexHull(self, traj, frame_index = -1):
        '''
        The purpose of this function is to calculate the gold surface area by the Convex Hull. We will use the default last frame for this computation.
        NOTE, we assume that your area does not change with trajectories. This is true if your gold atoms are frozen!
        INPUTS:
            self: class property
            traj: traj from md.traj
            frame_index: [int, default=-1] frame that you want the sasa for
        OUTPUTS:
            self.hull: [class object] hull object from ConvexHull
            self.surface_area_hull: [float] surface area based on ConvexHull in nm2
        '''
        ## IMPORING ConvexHull command
        from scipy.spatial import ConvexHull
        
        ## FINDING GOLD COORDINATES FOR LAST FRAME
        points = traj.xyz[frame_index, self.gold_index, : ]
        
        ## GETTING CONVEX HULL
        self.hull = ConvexHull(points = points)
        
        ## FINDING AREA
        self.surface_area_hull = self.hull.area
        
        return
    
    ### FUNCTION TO PLOT THE CONVEX HULL
    def plot_ConvexHull(self, want_all_plots=False):
        '''
        The purpose of this function is to plot gold atom with the convex hull.
        INPUTS:
            want_all_plots: [logical, default=False] True if you want all plots for ConvexHull
        OUTPUTS:
            Three plots:
                - figure of the gold points by itself (shown if want_all_plots = True)
                - figure of the convex hull by itself (shown if want_all_plots = True)
                - figure of the gold points with the convex hull
        '''
        plot_convex_hull(self.hull, want_all_plots = want_all_plots)
        return
    
    ### FUNCTION TO CALCULATE THE RDF OF A SINGLE FRAME
    def calc_rdf_gold_gold_single_frame(self, traj, frame_index=-1, r_range = (0.0, 2.0), bin_width=0.005, periodic=True, verbose = True ):
        '''
        The purpose of this function is to calculate the RDF for gold-gold in a single frame. We are assuming the gold atoms are more or less frozen. 
        Therefore, the RDF should correctly get the gold-gold atoms for a single frame.
        The resulting RDF can be used to gather information about coordination numbers, etc.
        INPUTS:
            self: class object
            traj: trajectory from md.traj
            frame_index: [int, default=-1] frame you are interested in using
            periodic: [logical, default=True] True if you want PBCs to be accounted for
            bin_width: [float, default = 0.005] bin width for the RDF
            r_range: [tuple, default= (0.0, 2.0)] range of interest for the RDF
            verbose: [logical, default=True] True if you want verbose information
        OUTPUTS:
            self.gold_gold_rdf_r: [np.array] radius vector for the RDF
            self.gold_gold_rdf_g_r: [np.array] g(r), RDF
        '''
        ## PRINTING
        if verbose == True:
            print("COMPUTING RDF BETWEEN GOLD-GOLD ATOMS")
        
        ## COMPUTING RDF FOR A SINGLE FRAME
        self.gold_gold_rdf_r, self.gold_gold_rdf_g_r = md.compute_rdf( traj = traj[frame_index],
                                                                        pairs = self.gold_gold_atom_pairs,
                                                                        r_range = r_range,
                                                                        bin_width = bin_width,
                                                                        periodic = periodic
                                                                        )
        
        return
    
    ### FUNCTION TO FIND THE FIRST SOLVATION SHELL/COORDINATION NUMBER
    def calc_rdf_first_solvation_shell( self, ):
        '''
        The purpose of this function is to get the first solvation shell of the gold-gold RDF
        INPUTS:
            self: class object
        OUTPUTS:
            self.gold_gold_rdf_first_solv_shell: [dict] contains information of the first solvation shell of the RDF, e.g.
                {'max': {'g_r': 1633.8344813198578, 'index': 55, 'r': 0.2775}, <-- maximum of the RDF
                 'min': {'g_r': 0.0, 'index': 61, 'r': 0.3075}} <-- minimum of the RDF
        '''
        self.gold_gold_rdf_first_solv_shell = find_first_solvation_shell_rdf(g_r = self.gold_gold_rdf_g_r,
                                                                             r = self.gold_gold_rdf_r
                                                                             )
        
        return
    
    ### FUNCTION TO PLOT THE RDF
    def plot_gold_gold_rdf(self):
        '''
        The purpose of this function is to plot the rdf between gold-gold pairs
        INPUTS:
            self: class object
        OUTPUTS:
            fig, ax of the RDF figure
        '''
        ## PLOTTING THE RDF
        fig, ax = plot_rdf(r = self.gold_gold_rdf_r,
                           g_r = self.gold_gold_rdf_g_r)
        ## SETTING TITLE
        ax.set_title("RDF between gold-gold atoms")
        ## DRAWING LABELS FOR MINIMA
        ax.axvline( x = self.gold_gold_rdf_first_solv_shell['min']['r'], color='r', linestyle='--', linewidth = 2, label = "First solvation min (%.2f nm)"%( self.gold_gold_rdf_first_solv_shell['min']['r']  ) )
        ## DRAWING LEGEND
        ax.legend()
        
        return fig, ax
    
    ### FUNCTION TO FIND GOLD COORDINATION NUMBERS
    def find_gold_gold_coordination_number(self, traj, cutoff, frame = -1, verbose = True):
        '''
        The purpose of this function is to find the gold-gold coordination numbers
        INPUTS:
            self: class object
            traj: trajectory from md.traj
            cutoff: [float]
                cutoff for coordination number of gold
            frame: [int, default = -1]
                frame that you want to run this coordination number calculation on. Note, we are assuming coordination number does not significantly change!
            verbose: [logical, default = True]
                True if you want verbose response
        OUTPUTS:
            gold_gold_coordination: [dict] dictionary containing information about gold-gold coordination, e.g.
                gold_coord_num: [np.array, shape=(num_gold, 2)] indices and coordination number of the gold atoms
                gold_gold_cutoff: [float] cutoff for gold-gold nearest neighbors calculation
                gold_surface_indices: [np.array] array with the indices of the surface atoms
                gold_surface_atom_index: [np.array] array with gold-surface atom indices (referenced to traj.topology)
                gold_surface_coord_num: [np.array, shape=(num_surface_atoms)] surface coordination numbers
                gold_surface_num_atom: [int] total number of surface atoms
                
            gold_gold_surface_facet_edge_dict: [dict] dictionary containing information about facet vs. edges
                gold_surface_facet_vs_edge_classification: [np.array, shape=(num_surface_atoms, 1)] Classification of edge and surface, where:
                    Values of 0 indicate edges
                    Values of 1 indicate facets
                gold_surface_facet_index:  [np.array, shape=(num_facets, 1)]   Indices of gold surface atoms that are faceted
                gold_surface_edge_index:   [np.array, shape=(num_edge, 1)]     Indices of gold surface atoms that are on the edge
        '''
        ## PRINTING
        if verbose == True:
            print("\n--- COMPUTING GOLD GOLD COORDINATION WITH CUTOFF OF %.3f ---"%(cutoff))
        ## FINDING COORDINATION NUMBER AND SURFACE ATOMS
        gold_gold_coordination = \
                    full_find_surface_atoms( traj                               = traj,
                                             cutoff                             = cutoff,
                                             gold_atom_index                    = self.gold_index,
                                             coord_num_surface_to_bulk_cutoff   = self.coord_num_surface_to_bulk_cutoff,
                                             frame = -1, # Use last frame for gold-gold distances
                                             verbose = verbose,
                                             periodic = True,
                                            )
        
        ## DETERMINE WHICH SURFACE ATOMS ARE EDGE AND FACET ATOMS
        gold_gold_surface_facet_edge_dict = \
                                find_facets_and_edges_based_on_coord_num( gold_gold_coordination = gold_gold_coordination,
                                                                          coord_num_facet_vs_edge_cutoff = self.coord_num_facet_vs_edge_cutoff,
                                                                          verbose = verbose
                                                                         )        
        return gold_gold_coordination, gold_gold_surface_facet_edge_dict
    
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
        coordinates_surface_facets  = traj.xyz[ frame, self.gold_gold_coordination['gold_surface_atom_index'][self.gold_gold_surface_facet_edge_dict['gold_surface_facet_index']], :]
        coordinates_surface_edges   = traj.xyz[ frame, self.gold_gold_coordination['gold_surface_atom_index'][self.gold_gold_surface_facet_edge_dict['gold_surface_edge_index']], :]
        
        ## PLOTTING THE FACETS AND EDGES
        # FACETS
        ax.scatter( coordinates_surface_facets[:, 0], coordinates_surface_facets[:, 1], coordinates_surface_facets[:, 2], **plot_dict['facet']  )
        # EDGES
        ax.scatter( coordinates_surface_edges[:, 0], coordinates_surface_edges[:, 1], coordinates_surface_edges[:, 2], **plot_dict['edge'] )
        
        ## ADDING LEGEND
        ax.legend(loc = 'lower right')
        
        return
#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON
    # analysis_dir=r"180328-ROT_SPHERICAL_TRANSFER_LIGANDS_6nm" # Analysis directory
    analysis_dir=r"SELF_ASSEMBLY_FULL" # Analysis directory
    # category_dir="spherical" # category directory
    category_dir="EAM" # category directory spherical
    # specific_dir="spherical_6_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    diameter="2"
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
    ### DEFINING INPUT DATA
    input_details={ 'gold_residue_name' : GOLD_RESIDUE_NAME,         # Gold residue name
                    'sulfur_atom_name'  : ATTACHED_SULFUR_ATOM_NAME, # Sulfur atom name
                    'gold_sulfur_cutoff': GOLD_SULFUR_CUTOFF,        # Gold sulfur cutoff for bonding
                    'gold_shape'        : category_dir,             # shape of the gold
                    'coord_num_surface_to_bulk_cutoff': 11,         # Cutoff between surfae and bulk
                    'coord_num_facet_vs_edge_cutoff':   7,              # Cutoff between facet and edge atoms
                    'ps_per_frame'      : 50,                        # Total picoseconds per frame
                    'split_traj'        : 200,                        # Total frames to run calculation every time
                    'gold_optimize'     : True,                     # True if you want gold optimization procedure
                    }
    
    ### FINDING SELF ASSEMBLY STRUCTURE
    structure = self_assembly_structure(traj_data, **input_details)
        
    #%%
    from MDDescriptors.application.nanoparticle.plot_self_assembly_structure import plot_self_assembly_structure
    
    plot_structure = plot_self_assembly_structure(structure)


    #%%
    ## PLOTTING 
    structure.plot_ConvexHull()

    #%%
    ## PLOTTING 3D
    structure.plot3d_surface_facets_vs_edges(traj=traj_data.traj)
    
    #%%
    
    structure.gold_gold_coordination['gold_surface_atom_index'][structure.gold_gold_surface_facet_edge_dict['gold_surface_facet_index']]