# -*- coding: utf-8 -*-
"""
nanoparticle_rdf.py
This script contains all RDF scripts for the nanoparticle project

CREATED ON: 06/27/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)

INPUTS:
    - gro file
    - itp file (for bonding purposes)
    
OUTPUTS:
    - rdf between heavy atom ligands to water
    
ALGORITHM:
    - get nanoparticle structure
    - get all heavy atom to oxygens of water pairs
    - calculate rdf using md.traj built in function
    
CLASSES:
    nanoparticle_rdf: computes nanoparticle RDFs
    
FUNCTIONS:
    mdtraj_rdf_custom: custom function of RDF that takes in distances rather than calculating distances
    plot_rdf: plots rdf
    calc_rdf_histogram_data_pieces: computes the rdf histogram in pieces
    compute_ranges_and_bins_for_rdf: computes ranges and bins for the RDF
    normalize_rdf: finds normalized rdfs
    calc_edges: finds edges for a histogram without having to use histogram method
    
**UPDATES**
- 20180628 - optimizing rdfs to deal with memory error
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
from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot


### FUNCTION TO CALCULATE RDF BASED ON DISTANCES
def mdtraj_rdf_custom(traj, distances, pairs, r_range=None, bin_width=0.005, n_bins=None, periodic=True, opt=True ):
    '''
    The purpose of this function is to generate an rdf based on mdtraj's function. Instead of computing distances, distances will be an input for this function.
    This is useful because sometimes distances cannot be computed in a general fashion as md.traj was designed for. 
    It has issues with large memories when atom_pairs and number of frames becomes too large!
    INPUTS:
        traj: trajectory
            trajectory from md.traj
        distances: array-like, shape=(n_frames,n_pairs)
            distances from md.compute_distances
        pairs: array-like, shape=(n_pairs, 2)
            atom pairs used for distance calculations
        r_range: array-like, shape=(2,), optional, default=(0.0, 1.0)
            Minimum and maximum radii.
        bin_width : float, optional, default=0.005
            Width of the bins in nanometers.
        n_bins : int, optional, default=None
            The number of bins. If specified, this will override the `bin_width`
            parameter.
        periodic : bool, default=True
            If `periodic` is True and the trajectory contains unitcell
            information, we will compute distances under the minimum image
            convention.
        opt : bool, default=True
            Use an optimized native library to compute the pair wise distances.
    OUTPUTS:
        r : np.ndarray, shape=(np.diff(r_range) / bin_width - 1), dtype=float
            Radii values corresponding to the centers of the bins.
        g_r : np.ndarray, shape=(np.diff(r_range) / bin_width - 1), dtype=float
            Radial distribution function values at r.
    '''
    ## COMPUTING RANGE AND BINS ACCORDING TO MDtraj
    if r_range is None:
        r_range = np.array([0.0, 1.0])
    if n_bins is not None:
        n_bins = int(n_bins)
        if n_bins <= 0:
            raise ValueError('`n_bins` must be a positive integer')
    else:
        n_bins = int((r_range[1] - r_range[0]) / bin_width)
    
    ## COMPUTING HISTOGRAM INFORMATION
    g_r, edges = np.histogram(distances, range=r_range, bins=n_bins)
    r = 0.5 * (edges[1:] + edges[:-1])
    ## COMPUTING RDF AND NORMALIZING BASED ON MDTRAJ
    # Normalize by volume of the spherical shell.
    # See discussion https://github.com/mdtraj/mdtraj/pull/724. There might be
    # a less biased way to accomplish this. The conclusion was that this could
    # be interesting to try, but is likely not hugely consequential. This method
    # of doing the calculations matches the implementation in other packages like
    # AmberTools' cpptraj and gromacs g_rdf.
    V = (4 / 3) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    norm = len(pairs) * np.sum(1.0 / traj.unitcell_volumes) * V
    g_r = g_r.astype(np.float64) / norm  # From int64.
    
    return r, g_r

### FUNCTION TO CALCULATE HISTOGRAM DATA IN PIECES
def calc_rdf_histogram_data_pieces(traj, atom_pairs, periodic, r_range, n_bins):
    '''
    The purpose of this function is to calculate the histogram data in pieces to ensure that the distance matrix does not cause MemoryError
    INPUTS:
        traj: trajectory from md.traj
        periodic: [logical, default=True] True if you want PBCs to be accounted for
        atom_pairs: [list] atom pairs for your system
        r_range: array-like, shape=(2,), optional, default=(0.0, 1.0)
            Minimum and maximum radii.
        n_bins : int, optional, default=None
            The number of bins. If specified, this will override the `bin_width`
            parameter.
    OUTPUTS:
        hist: [np.array] histogram_data or equivalently, the number of occurances for specific bins
            NOTE: hist is returned as an array within a list to prevent concatentation of numpy from messing things up!
    '''
    ## CALCULATING DISTANCES
    distances = md.compute_distances( traj = traj, atom_pairs = atom_pairs, periodic = periodic, opt = True )
    ## FINDING HISTOGRAM DATA
    hist, edges = np.histogram(distances, range=r_range, bins=n_bins)
    return [hist]

### FUNCTION TO COMPUTE RANGES AND NUMBER OF BINS
def compute_ranges_and_bins_for_rdf( r_range=None, bin_width=0.005, n_bins=None ):
    '''
    The purpose of this function is to compute the number of bins / r range for radial distribution functions
    INPUTS:
        r_range: array-like, shape=(2,), optional, default=(0.0, 1.0)
            Minimum and maximum radii.
        n_bins : int, optional, default=None
            The number of bins. If specified, this will override the `bin_width`
            parameter.
        bin_width : float, optional, default=0.005
            Width of the bins in nanometers.
    OUTPUTS:
        r_range: array-like, shape=(2,), optional, default=(0.0, 1.0)
            Minimum and maximum radii.
        n_bins : int
            The number of bins. If specified, this will override the `bin_width`
            parameter.
    '''
    ## COMPUTING RANGE AND BINS ACCORDING TO MDtraj
    if r_range is None:
        r_range = np.array([0.0, 1.0])
    if n_bins is not None:
        n_bins = int(n_bins)
        if n_bins <= 0:
            raise ValueError('`n_bins` must be a positive integer')
    else:
        n_bins = int((r_range[1] - r_range[0]) / bin_width)
    return r_range, n_bins

### FUNCTION TO NORMALIZE THE RDF
def normalize_rdf(traj, g_r, edges, atom_pairs ):
    '''
    The purpose of this function is to normalize the RDF given the histogram data
    INPUTS:
        traj: trajectory from md.traj
        g_r: [np.array] histogram data
        edges: [np.array] bin edges from np.histogram
        atom_pairs: [list] atom pairs for your system
    OUTPUTS:
        r : np.ndarray, shape=(np.diff(r_range) / bin_width - 1), dtype=float
            Radii values corresponding to the centers of the bins.
        g_r : np.ndarray, shape=(np.diff(r_range) / bin_width - 1), dtype=float
            Radial distribution function values at r.
    '''
    ## COMPUTING HISTOGRAM INFORMATION
    r = 0.5 * (edges[1:] + edges[:-1])
    ## COMPUTING RDF AND NORMALIZING BASED ON MDTRAJ
    # Normalize by volume of the spherical shell.
    # See discussion https://github.com/mdtraj/mdtraj/pull/724. There might be
    # a less biased way to accomplish this. The conclusion was that this could
    # be interesting to try, but is likely not hugely consequential. This method
    # of doing the calculations matches the implementation in other packages like
    # AmberTools' cpptraj and gromacs g_rdf.
    V = (4 / 3) * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    norm = len(atom_pairs) * np.sum(1.0 / traj.unitcell_volumes) * V
    g_r = g_r.astype(np.float64) / norm  # From int64.
    return r, g_r
    
### FUNCTION TO GET THE EDGES OF THE HISTOGRAM BASED ON R_RANGE AND NUM_BIN
def calc_edges(r_range, n_bins):
    '''
    The purpose of this function is to find the edges, as outputted by np.histogram, without running the histogram function
    INPUTS:
        r_range: array-like, shape=(2,), optional, default=(0.0, 1.0)
            Minimum and maximum radii.
        n_bins : int, optional, default=None
            The number of bins. If specified, this will override the `bin_width`
            parameter.
    OUTPUTS:
        edges: [np.array, shape=(N,1)] edges for your histogram
    '''
    return np.linspace(start=r_range[0], stop=r_range[1], num=n_bins+1) # Number of bins need to be added by 1 to account for end points


###################################
### CLASS TO CALCULATE THE RDFs ###
###################################
class nanoparticle_rdf:
    '''
    The purpose of this class is to calculate the radial distribution function (RDF) of the nanoparticle ligands to surrounding environment
    INPUTS:
        traj_data: Data taken from import_traj class
        ligand_names: [list] list of ligand residue names. Note that you can list as much as you'd like, we will check with the trajectory to ensure that it is actually there.
        water_residue_name: [str, default='HOH'] residue name of water
        itp_file: itp file name (note! We will look for the itp file that should be in the same directory as your trajectory)
            if 'match', it will look for all itp files within the trajectory directory and locate all itp files that match your ligand names.
                NOTE: This is currently assuming homogenous ligands -- this will need to be adjusted for multiple ligands
        bin_width: [float, default = 0.2] bin width in nm of the RDF
        save_disk_space: [logical, Default: True] True if you want to save disk space by removing the following variables
        r_range: [tuple, shape=2] range to plot your RDF
        split_traj: [int] number of frames to split the trajectoyr
    OUTPUTS:
        ## STORED INPUTS
        self.bin_width, self.water_residue_name, self.r_range
        ## STRUCTURE
        self.structure_np: structure from nanoparticle_structure class
    FUNCTIONS:
        clean_disk: cleans up disk
        calc_rdf_lig_heavy_atoms_to_water: calculates RDF between the last heavy atom to water
    '''
    ### INITIALIZING
    def __init__(self, traj_data, ligand_names, itp_file, split_traj = 50, separated_ligands = False, bin_width = 0.2, r_range=(0.0, 2.0),  water_residue_name = 'HOH', save_disk_space = True):
        
        ## STORING INPUTS
        self.bin_width = bin_width
        self.water_residue_name = water_residue_name
        self.r_range = r_range
        
        
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__))
        
        ### CALCULATING NANOPARTICLE STRUCTURE
        self.structure_np = nanoparticle_structure(traj_data           = traj_data,                # trajectory data
                                                ligand_names        = ligand_names,        # ligand names
                                                itp_file            = itp_file,                 # defines the itp file
                                                structure_types      = None,                     # checks structural types
                                                separated_ligands    = separated_ligands    # True if you want separated ligands 
                                                )
        ## DEFINING THE TRAJECTORY
        traj = traj_data.traj # [0:1000]
        
        ## FINDING THE RDF BETWEEN HEAVY ATOM TO WATER
        self.rdf_lig_heavy_atoms_to_water_r, self.rdf_lig_heavy_atoms_to_water_g_r  = self.calc_rdf_lig_heavy_atoms_to_water_optimized(traj, split_traj = split_traj) # 
        
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
            for each_item in [ #self.distance_ligand_heavy_atoms_to_water, 
                              self.ligand_heavy_atom_index_flatten,
                              self.hist, self.edges]:
                try:
                    each_item = []
                except Exception:
                    pass            
        return
    
    ### FUNCTION TO CALCULATE RDF BETWEEN HEAVY ATOMS AND WATER
    def calc_rdf_lig_heavy_atoms_to_water(self, traj, split_traj=50, periodic = True):
        '''
        The purpose of this function is to calculate the RDF between the heavy atoms to water (oxygens).
        **DEPRECIATED**
        INPUTS:
            traj: trajectory from md.traj
            periodic: [logical, default=True] True if you want PBCs to be accounted for
        OUTPUTS:
            self.ligand_heavy_atom_index: [np.array] ligand atom index (last ones)
            self.ligand_heavy_atom_index_flatten: [np.array, shape=(num_atoms, 1)] heavy atom flattened out
            self.distance_ligand_heavy_atoms_to_water: [np.array, shape=(num_frames, num_heavy_atoms, num_water)] distance matrix between heavy atoms and water
        '''
        ## DEFINING LIGAND HEAVY ATOM INDEX (FLATTEN)
        self.ligand_heavy_atom_index = np.array(self.structure_np.ligand_heavy_atom_index)[:,-1] ## FINDS LAST ATOM INDEX
        self.ligand_heavy_atom_index_flatten = np.array(self.ligand_heavy_atom_index).flatten()
        
        ## FINDING WATER INDEXES
        self.num_water_residues, self.water_residue_index, self.water_oxygen_index = calc_tools.find_water_index(traj = traj, water_residue_name = self.water_residue_name)
        
        ## CREATING ATOM PAIRS BETWEEN LIGAND AND WATER
        atom_pairs_lig_water = calc_tools.create_atom_pairs_list(atom_1_index_list = self.ligand_heavy_atom_index_flatten, 
                                                                      atom_2_index_list = self.water_oxygen_index)
        
        ## DEFINING INPUTS
        distances_input = {
                            'atom_pairs'    : atom_pairs_lig_water,
                            'periodic'      : periodic,
                            }
        
        ## FINDING DISTANCE MATRIX BASED ON THE ATOM PAIRS
        self.distance_ligand_heavy_atoms_to_water = calc_tools.split_traj_function( traj = traj,
                                                                             split_traj = split_traj,
                                                                             input_function = md.compute_distances,
                                                                             optimize_memory = True,
                                                                             **distances_input )
        
        ## RUNNING RDF FUNCTION
        r, g_r = mdtraj_rdf_custom( traj = traj,
                                    distances = self.distance_ligand_heavy_atoms_to_water,
                                    pairs = atom_pairs_lig_water,
                                    bin_width = self.bin_width,
                                    r_range = self.r_range, 
                                    periodic = periodic,
                                   )

        return  r, g_r
    
    ### FUNCTION TO CALCULATE RDF BETWEEN HEAVY ATOMS AND WATER
    def calc_rdf_lig_heavy_atoms_to_water_optimized(self, traj, split_traj=50, periodic = True):
        '''
        The purpose of this function is to calculate the RDF between the heavy atoms to water (oxygens).
        NOTE: This function corrects for memory error!
        INPUTS:
            traj: trajectory from md.traj
            periodic: [logical, default=True] True if you want PBCs to be accounted for
        OUTPUTS:
            self.ligand_heavy_atom_index: [np.array] ligand atom index (last ones)
            self.ligand_heavy_atom_index_flatten: [np.array, shape=(num_atoms, 1)] heavy atom flattened out
            self.distance_ligand_heavy_atoms_to_water: [np.array, shape=(num_frames, num_heavy_atoms, num_water)] distance matrix between heavy atoms and water
        '''
        ## DEFINING LIGAND HEAVY ATOM INDEX (FLATTEN)
        self.ligand_heavy_atom_index = np.array(self.structure_np.ligand_heavy_atom_index)[:,-1] ## FINDS LAST ATOM INDEX
        self.ligand_heavy_atom_index_flatten = np.array(self.ligand_heavy_atom_index).flatten()
        
        ## FINDING WATER INDEXES
        self.num_water_residues, self.water_residue_index, self.water_oxygen_index = calc_tools.find_water_index(traj = traj, water_residue_name = self.water_residue_name)
        
        ## CREATING ATOM PAIRS BETWEEN LIGAND AND WATER
        atom_pairs_lig_water = calc_tools.create_atom_pairs_list(atom_1_index_list = self.ligand_heavy_atom_index_flatten, 
                                                                      atom_2_index_list = self.water_oxygen_index)
        
        ## CALCULATING RANGES AND BINS
        r_range, n_bins = compute_ranges_and_bins_for_rdf( r_range = self.r_range, bin_width = self.bin_width  )
        
        ## DEFINING INPUTS
        split_input = {
                            'atom_pairs'    : atom_pairs_lig_water, # Atom pairs
                            'periodic'      : periodic,             # Periodicity
                            'r_range'       : r_range,              # Range between RDFs
                            'n_bins'        : n_bins,               # Number of bins for RDF
                            }
        
        ## FINDING DISTANCE MATRIX BASED ON THE ATOM PAIRS
        self.hist = calc_tools.split_traj_function( traj = traj,
                                                     split_traj = split_traj,
                                                     input_function = calc_rdf_histogram_data_pieces,
                                                     optimize_memory = False, # False because this does not take too much memory!
                                                     **split_input )

        ## SUMMING UP THE HISTOGRAMS
        self.hist = np.sum(self.hist, axis = 0)
        
        ## FINDING EDGES
        self.edges = calc_edges( r_range = r_range, n_bins = n_bins )
        
        ## NORMALIZING TO GET RDF
        r, g_r = normalize_rdf( traj = traj, g_r = self.hist, edges = self.edges, atom_pairs = atom_pairs_lig_water  )

        return  r, g_r
    


### FUNCTION TO PLOT RDF
def plot_rdf(r, g_r, label = None):
    '''
    The purpose of this function is to plot the rdf
    INPUTS:
        r: [np.array] r value for g(r)
        g_r: [np.array] radial distribution function
        label: [str] label for the RDF
    OUTPUTS:
        fig, ax -- figure, axis for the plot
    '''
    ### IMPORTING MODULES
    import matplotlib.pyplot as plt
    ## DEFAULT PLOTTING STYLES
    from MDDescriptors.global_vars.plotting_global_vars import FONT_SIZE, FONT_NAME, COLOR_LIST, LINE_STYLE, DPI_LEVEL
    ## CREATING PLOT
    fig = plt.figure() 
    ax = fig.add_subplot(111)

    ## DRAWING LABELS
    ax.set_xlabel('r (nm)',fontname=FONT_NAME,fontsize = FONT_SIZE)
    ax.set_ylabel('Radial Distribution Function',fontname=FONT_NAME,fontsize = FONT_SIZE)
    
    # Drawing ideal gas line
    ax.axhline(y=1, linewidth=1, color='black', linestyle='--')
    
    ## PLOTTING RDF
    ax.plot(r, g_r, '-', color = 'k',
            label= label, **LINE_STYLE)
    return fig, ax





#%% MAIN SCRIPT
if __name__ == "__main__":
    
    
#    ### DIRECTORY TO WORK ON    
#    analysis_dir=r"180607-Alkanethiol_Extended_sims" # Analysis directory
#    category_dir="EAM" # category directory
#    specific_dir="EAM_310.15_K_2_nmDIAM_butanethiol_CHARMM36_Trial_1" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
#    
#    ### DEFINING FULL PATH TO WORKING DIRECTORY
#    path2AnalysisDir=r"R:\scratch\nanoparticle_project\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir + '\\' # PC Side
    
    
    ## DEFINING SIMULATION PATH
    path_sim=r"R:\scratch\nanoparticle_project\simulations\HYDROPHOBICITY_PROJECT_ROTELLO\EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_Trial_1"
    
    ### DEFINING FILE NAMES
    gro_file=r"sam_prod.gro" # Structural file
    xtc_file=r"sam_prod_10_ns_whole.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file


    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path_sim, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    
    
    #%%
    ### DEFINING INPUT DATA
    input_details = {   'traj_data'          :           traj_data,                      # Trajectory information
                         'ligand_names'      :           ['OCT', 'BUT', 'HED', 'DEC', 'DOD', 'R12'],   # Name of the ligands of interest
                         'itp_file'          :           'sam.itp',                      # ITP FILE
                         'bin_width'         :          0.02,                                # Bin width   
                         'r_range'           :          (0.0, 10.0),                      # r range to run the rdf function
                         'save_disk_space'   :          True    ,                        # Saving space
                         'split_traj'        :          50, # Number of frames to split trajectory
                         'separated_ligands' :          False,
                         }
    ### CALCULATING WATER STRUCTURE
    rdf = nanoparticle_rdf( **input_details )
    
    #%%
    
    plot_rdf(r = rdf.rdf_lig_heavy_atoms_to_water_r,
             g_r = rdf.rdf_lig_heavy_atoms_to_water_g_r)
    
    ## IMPORTING TOOLS
    from MDDescriptors.geometry.rdf_extract import find_first_solvation_shell_rdf
    
    ## FINDING FIRST SOLVATION SHELL
    first_solv_shell = find_first_solvation_shell_rdf(g_r = rdf.rdf_lig_heavy_atoms_to_water_g_r,
                                                      r = rdf.rdf_lig_heavy_atoms_to_water_r)
    
    #%%
    n_bins=None
    bin_width=0.02
    r_range=(0.0, 3.0)
    
    

    
    
    distance_matrix = rdf.distance_ligand_heavy_atoms_to_water[0:20, :]
    
    ## FINDING HISTOGRAM
    g_r, edges = np.histogram(distance_matrix, range=r_range, bins=n_bins)
    ## RETURNS: 
    #   - g_r is the number of occurances within a bin
    #   - edges: array containing the bins (e.g. 0.0, 0.02, ...)
    
    distance_matrix = rdf.distance_ligand_heavy_atoms_to_water[20:40, :]
    
    ## FINDING HISTOGRAM
    g_r_2, edges = np.histogram(distance_matrix, range=r_range, bins=n_bins)
    
    test_g_r = g_r + g_r_2
    
    
#%%
    ## CREATING ATOM PAIRS BETWEEN LIGAND AND WATER
    atom_pairs_lig_water = calc_tools.create_atom_pairs_list(atom_1_index_list = rdf.ligand_heavy_atom_index_flatten, 
                                                                  atom_2_index_list = rdf.water_oxygen_index)
    
    hist = rdf.calc_rdf_histogram_data_pieces(traj=traj_data.traj[0:20], atom_pairs = atom_pairs_lig_water, periodic = True, r_range = rdf.r_range, n_bins = n_bins )
    
    
    
    